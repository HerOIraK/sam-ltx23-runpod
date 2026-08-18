#!/usr/bin/env python3
"""Generate a pip constraints file that freezes the ABI-critical packages.

Why this exists
---------------
SageAttention, comfy_kitchen and other compiled extensions link against the
exact PyTorch C++ ABI they were built with. If anything later installs a
different torch, those .so files stop loading with errors like:

    ImportError: .../sageattention/_fused...so: undefined symbol:
    _ZNK3c1010TensorImpl15decref_pyobjectEv

The usual culprit is a custom node installed at runtime through
ComfyUI-Manager, whose requirements.txt names a torch version. Build-time
requirement filtering cannot stop that, because it happens on the live pod.

Pointing PIP_CONSTRAINT at the file this script produces makes every later
pip invocation -- including the ones ComfyUI-Manager runs -- unable to move
these packages. pip honours PIP_CONSTRAINT globally, with no cooperation
needed from the caller.

Usage
-----
    python3 make-pip-constraints.py [output-path]

Default output path is /etc/pip-constraints.txt.
Always exits 0. A diagnostic must never fail a build.
"""

import sys

# Packages whose version must never change after the image is built.
# torch first: everything else here is downstream of its ABI.
PINNED = [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton",
    "sageattention",
    "xformers",
    "flash-attn",
]


def installed_version(name):
    """Return the installed version string, or None if absent/unreadable."""
    try:
        import importlib.metadata as md
    except Exception:
        return None
    try:
        return md.version(name)
    except Exception:
        return None


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/etc/pip-constraints.txt"

    lines = [
        "# Generated at image build time by make-pip-constraints.py",
        "# Do not edit by hand. Referenced via PIP_CONSTRAINT.",
        "# These packages are frozen because compiled extensions in this",
        "# image link against their exact ABI.",
    ]

    found = 0
    for name in PINNED:
        version = installed_version(name)
        if version is None:
            continue
        # A constraint only takes effect if the package is actually requested
        # by something. Listing an absent package is harmless but pointless.
        lines.append("{}=={}".format(name, version))
        found += 1

    if found == 0:
        lines.append("# WARNING: none of the pinned packages were found")

    body = "\n".join(lines) + "\n"

    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
    except Exception as exc:
        sys.stderr.write(
            "make-pip-constraints: could not write {}: {}\n".format(out_path, exc)
        )
        sys.stdout.write(body)
        return 0

    sys.stdout.write(body)
    sys.stdout.write(
        "make-pip-constraints: wrote {} pin(s) to {}\n".format(found, out_path)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # never fail the build
        sys.stderr.write("make-pip-constraints: unexpected error: {}\n".format(exc))
        sys.exit(0)
