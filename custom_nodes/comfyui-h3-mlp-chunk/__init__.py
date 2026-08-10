"""
comfyui-h3-mlp-chunk
Caps peak activation memory in the MiniMax H3 text encoder by running its MLP
in sequence-length chunks instead of one giant matmul.

Why this exists
---------------
MiniMax H3's text encoder is a Qwen3-VL truncated to 50 layers. Every decoder
layer's MLP computes, on one line:
    down_proj(activation(gate_proj(x)) * up_proj(x))
With hidden=5120 and intermediate=25600, that materialises two [tokens, 25600]
bf16 tensors at the same time before down_proj consumes them.

For a text prompt of a few hundred tokens this is nothing. For ref2va with a
reference *video*, the vision tower emits tens of thousands of tokens and the
same line allocates gigabytes, per layer:
     500 tokens -> ~0.07 GB
    2000 tokens -> ~0.29 GB
   12000 tokens -> ~1.72 GB
   60000 tokens -> ~8.58 GB <-- OOM on a 24 GB card

Splitting the sequence dimension makes peak activation roughly flat at ~0.15 GB
regardless of token count. Rows of the sequence do not interact inside the
MLP, so this is a pure loop-and-concatenate over independent work, not an
approximation. Attention is untouched.

Configuration
-------------
H3_MLP_CHUNK=1024       tokens per chunk (default)
H3_MLP_CHUNK=0          disable the patch entirely

This module registers no ComfyUI nodes. It patches on import and gets out of the way.
"""
import inspect
import os

# ComfyUI requires these to exist. This pack deliberately adds no nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_TAG = "[h3-mlp-chunk]"
_TARGET_MODULE = "comfy.text_encoders.llama"


def _chunk_size():
    raw = os.environ.get("H3_MLP_CHUNK", "1024")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        print("{} H3_MLP_CHUNK={!r} is not an integer; falling back to 1024".format(_TAG, raw))
        return 1024


def _find_mlp_classes(module):
    """
    Locate gate/up/down MLP classes by reading the source of their forward().
    Matching structurally rather than by class name means an upstream rename does
    not silently turn this patch into a no-op, and it avoids touching attention
    or norm classes that happen to live in the same module.
    """
    found = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if getattr(obj, "_h3_chunk_patched", False):
            # Already wrapped. The wrapper's source no longer mentions the
            # projection names, so recognise it explicitly instead of
            # reporting "nothing found" on a rescan.
            found.append((name, obj))
            continue
        fwd = obj.__dict__.get("forward")
        if fwd is None:
            continue
        try:
            src = inspect.getsource(fwd)
        except (OSError, TypeError):
            continue
        if "gate_proj" in src and "up_proj" in src and "down_proj" in src:
            found.append((name, obj))
    return found


def _make_chunked_forward(original_forward, chunk):
    def forward(self, x, *args, **kwargs):
        # Short sequences take the original path untouched: no copy, no concat,
        # no behaviour change for ordinary text prompts.
        try:
            too_small = x.dim() < 2 or x.shape[-2] <= chunk
        except AttributeError:
            return original_forward(self, x, *args, **kwargs)
        if too_small:
            return original_forward(self, x, *args, **kwargs)

        import torch
        parts = []
        for i in range(0, x.shape[-2], chunk):
            parts.append(original_forward(self, x[..., i:i + chunk, :], *args, **kwargs))
        return torch.cat(parts, dim=-2)

    forward.__name__ = getattr(original_forward, "__name__", "forward")
    forward.__doc__ = getattr(original_forward, "__doc__", None)
    return forward


def _install():
    chunk = _chunk_size()
    if chunk == 0:
        print("{} disabled (H3_MLP_CHUNK=0)".format(_TAG))
        return
    import importlib
    module = importlib.import_module(_TARGET_MODULE)
    classes = _find_mlp_classes(module)
    if not classes:
        print("{} no gate/up/down MLP class found in {}; nothing patched".format(_TAG, _TARGET_MODULE))
        return
    patched = []
    for name, cls in classes:
        if getattr(cls, "_h3_chunk_patched", False):
            continue
        cls.forward = _make_chunked_forward(cls.__dict__["forward"], chunk)
        cls._h3_chunk_patched = True
        patched.append(name)
    if patched:
        print("{} chunk={} tokens; patched {}".format(
            _TAG, chunk, ", ".join("{}.{}".format(_TARGET_MODULE, n) for n in patched)))
    else:
        print("{} already patched; nothing to do".format(_TAG))


try:
    _install()
except Exception as exc:  # noqa: BLE001 - a memory optimisation must never break startup
    print("{} patch failed, continuing unpatched: {}: {}".format(_TAG, type(exc).__name__, exc))
