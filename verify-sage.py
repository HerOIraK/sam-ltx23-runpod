#!/usr/bin/env python3
"""
SageAttention build-time and runtime verifier.

Build time (no GPU visible):
    Proves the installed wheel actually contains the compiled extension modules
    for every target architecture.

Runtime (GPU visible):
    Additionally runs real forward passes through the kernels this GPU will use.

--------------------------------------------------------------------------
WHY THE OLD CHECK COULD NEVER PASS
--------------------------------------------------------------------------
sageattention/core.py defines ONLY these three flags:

    SM80_ENABLED   SM89_ENABLED   SM90_ENABLED

There is NO SM86_ENABLED anywhere upstream. Asserting on it makes
getattr(core, "SM86_ENABLED", None) return None forever, so the gate is
unsatisfiable regardless of how the wheel was compiled.

The RTX 3090 (sm_86) is served by the sm80 extension, and setup.py builds
_qattn_sm80 whenever HAS_SM80 or HAS_SM86 or HAS_SM89 ... is true.

Runtime dispatch inside sageattn() is:
    sm80 -> sageattn_qk_int8_pv_fp16_cuda   (CUDA ext)
    sm86 -> sageattn_qk_int8_pv_fp16_triton (Triton! RTX 3090)
    sm89 -> sageattn_qk_int8_pv_fp8_cuda    (needs _qattn_sm89, RTX 4090)
"""
import glob
import os
import sys


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


import torch
import sageattention
from sageattention import core

try:
    from importlib.metadata import version
    sage_ver = version("sageattention")
except Exception:
    sage_ver = getattr(sageattention, "__version__", "unknown")

print(f"torch           : {torch.__version__} | cuda {torch.version.cuda}")
print(f"sageattention   : {sage_ver}")

# Real fallback detection. hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda")
# is useless here: that Python function always exists and only asserts when
# it is actually called.
if not str(sage_ver).startswith("2."):
    fail(f"expected SageAttention 2.x, got {sage_ver} -- pip fell back to an older release")

pkg_dir = os.path.dirname(sageattention.__file__)
sos = sorted(os.path.basename(p) for p in glob.glob(os.path.join(pkg_dir, "*.so")))
print(f"extensions      : {sos or 'NONE'}")

if not any("sm80" in s for s in sos):
    fail("_qattn_sm80 not built -> RTX 3090 (sm_86) INT8 CUDA path missing")
if not any("sm89" in s for s in sos):
    fail(
        "_qattn_sm89 not built -> RTX 4090 (sm_89) FP8 path missing.\n"
        "       SageAttention's setup.py splits TORCH_CUDA_ARCH_LIST on ';' and ',' ONLY.\n"
        "       A space-separated value like '8.6 8.9' is parsed as ONE capability,\n"
        "       matches startswith('8.6'), and silently drops sm_89.\n"
        "       Use '8.6;8.9'."
    )
if not any(s.startswith("_fused") for s in sos):
    fail("_fused not built -> quantisation kernels missing")

print(
    f"core flags      : SM80_ENABLED={core.SM80_ENABLED} "
    f"SM89_ENABLED={core.SM89_ENABLED} SM90_ENABLED={core.SM90_ENABLED}"
)
print("                  (SM86_ENABLED does not exist upstream -- do not assert on it)")

if not core.SM80_ENABLED:
    fail("_qattn_sm80 is on disk but failed to import")
if not core.SM89_ENABLED:
    fail("_qattn_sm89 is on disk but failed to import")

try:
    import triton
    print(f"triton          : {triton.__version__}  (RTX 3090 default path uses it)")
except Exception as exc:
    print(f"triton          : MISSING -> {exc}")
    print("                  On sm_86, sageattn() dispatches to the Triton kernel.")

if not torch.cuda.is_available():
    print("GPU             : none visible -- static verification PASSED")
    sys.exit(0)

cap = torch.cuda.get_device_capability(0)
sm = cap[0] * 10 + cap[1]
print(f"GPU             : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")

q = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="cuda")
k, v = q.clone(), q.clone()

o = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
torch.cuda.synchronize()
if not bool(torch.isfinite(o).all()):
    fail("sageattn() produced non-finite output")
print(f"sageattn()      : {tuple(o.shape)} finite=True")

if sm >= 80:
    o2 = sageattention.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, tensor_layout="HND", pv_accum_dtype="fp32"
    )
    torch.cuda.synchronize()
    print(f"sm80 int8+fp16  : OK finite={bool(torch.isfinite(o2).all())}")

if sm >= 89:
    o3 = sageattention.sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, tensor_layout="HND", pv_accum_dtype="fp32+fp16"
    )
    torch.cuda.synchronize()
    print(f"sm89 int8+fp8   : OK finite={bool(torch.isfinite(o3).all())}")
else:
    print("sm89 int8+fp8   : skipped -- no FP8 tensor cores on this GPU")
    print("                  Patch Sage Attention KJ -> auto or sageattn_qk_int8_pv_fp16_cuda")

print("SageAttention verification PASSED")
