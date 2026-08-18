#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ComfyUI version gate.
# ---------------------------------------------------------------------------
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
MIN_VERSION="${COMFYUI_MIN_VERSION:-0.33.0}"
PIN_REF="${COMFYUI_REF:-}"

echo "=============================================================="
echo " ComfyUI version gate"
echo "=============================================================="

cd "$COMFY_DIR"
git config --global --add safe.directory '*' 2>/dev/null || true

read_version() {
    python3 - <<'PY'
import re, pathlib, sys
p = pathlib.Path("comfyui_version.py")
if p.exists():
    m = re.search(r"""__version__\s*=\s*['\"]([^'\"]+)['\"]""", p.read_text())
    if m:
        print(m.group(1)); sys.exit(0)
p = pathlib.Path("pyproject.toml")
if p.exists():
    m = re.search(r"""(?m)^\s*version\s*=\s*['\"]([^'\"]+)['\"]""", p.read_text())
    if m:
        print(m.group(1)); sys.exit(0)
print("0.0.0")
PY
}

ver_ge() {  # ver_ge A B -> exit 0 when A >= B
    python3 - "$1" "$2" <<'PY'
import re, sys
def t(v):
    n = [int(x) for x in re.findall(r"\d+", v)[:3]]
    return tuple(n + [0] * (3 - len(n)))
sys.exit(0 if t(sys.argv[1]) >= t(sys.argv[2]) else 1)
PY
}

BAKED="$(read_version)"
echo "version file: ${BAKED}"
echo "required min: ${MIN_VERSION}"
echo "explicit pin: ${PIN_REF:-<none - verify only>}"
echo

# --------------------------------------------------- realign pinned helper pkgs
if [ -f requirements.txt ]; then
    echo "realigning helper packages to this revision's pins..."
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"
    python3 /usr/local/bin/filter-req.py requirements.txt /tmp/comfy-req.filtered
    # Install dependencies with --no-deps first, then full pass without changing torch
    pip install --no-cache-dir --no-deps -r /tmp/comfy-req.filtered || true
    pip install --no-cache-dir -c /etc/pip-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu130 -r /tmp/comfy-req.filtered || true
    TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
    echo "torch: ${TORCH_BEFORE} -> ${TORCH_AFTER}"
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
        echo "WARN: torch moved, restoring 2.10.0+cu130..."
        pip install --no-cache-dir --force-reinstall "torch==2.10.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
    fi
fi

FINAL_VERSION="$(read_version)"
echo "final ComfyUI version: ${FINAL_VERSION}"

if ! ver_ge "$FINAL_VERSION" "$MIN_VERSION"; then
    echo "FATAL: ComfyUI ${FINAL_VERSION} is below minimum ${MIN_VERSION}."
    exit 1
fi

echo "gate        : ${FINAL_VERSION} >= ${MIN_VERSION} OK"
echo "=============================================================="
