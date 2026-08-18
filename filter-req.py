#!/usr/bin/env python3
"""
Strip core-runtime pins out of a requirements.txt before installing it.

Usage:  filter-req.py <input.txt> <output.txt>
"""
import pathlib
import re
import sys

# Package names that must never be touched by a custom node's requirements.
# Normalised to lowercase with underscores turned into hyphens (PEP 503).
BLOCKED = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchsde",
    "triton",
    "pytorch-triton",
    "xformers",
    "flash-attn",
    "sageattention",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "transformers",
    "tokenizers",
    "accelerate",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
}

# pip directives that can redirect installs to a different wheel index.
BLOCKED_DIRECTIVES = (
    "--index-url",
    "--extra-index-url",
    "-i ",
    "--find-links",
    "-f ",
    "--trusted-host",
    "--pre",
)


def requirement_name(raw_line):
    """Return the normalised distribution name, '__DIRECTIVE__', or None."""
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    lowered = line.lower()
    if lowered.startswith(BLOCKED_DIRECTIVES):
        return "__DIRECTIVE__"
    if line.startswith("-"):
        return None

    line = line.split("#", 1)[0].strip()
    line = line.split(";", 1)[0].strip()
    if not line:
        return None

    if "@" in line and not line.startswith("@"):
        line = line.split("@", 1)[0].strip()

    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    if not match:
        return None
    return match.group(1).lower().replace("_", "-").replace(".", "-")


def main():
    if len(sys.argv) != 3:
        print("usage: filter-req.py <input.txt> <output.txt>", file=sys.stderr)
        return 2

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])

    kept, dropped = [], []
    for raw in src.read_text(encoding="utf-8", errors="replace").splitlines():
        name = requirement_name(raw)
        if name == "__DIRECTIVE__" or (name and name in BLOCKED):
            dropped.append(raw.strip())
        else:
            kept.append(raw)

    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")

    for line in dropped:
        print("    DROPPED: %s" % line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
