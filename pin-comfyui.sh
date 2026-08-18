#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ComfyUI version gate.
#
# Reports the current state BEFORE changing anything, then:
#   - COMFYUI_REF empty  -> verify only. Pass if baked version >= minimum.
#                           Never mutates the repo. This is the default.
#   - COMFYUI_REF set    -> move the checkout to that ref, with correct
#                           shallow-tag refspecs and escalating fallbacks.
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

# ---------------------------------------------------------------- diagnostics
echo "--- current state (read-only) ---"
if [ -d .git ]; then
    echo "git repo    : yes"
    echo "remote(s)   :"
    git remote -v 2>/dev/null | sed 's/^/              /' || echo "              (none)"
    echo "HEAD        : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "describe    : $(git describe --tags --always 2>/dev/null || echo none)"
    echo "branch      : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
    if [ -f .git/shallow ]; then echo "shallow     : YES"; else echo "shallow     : no"; fi
    echo "fetch spec  : $(git config --get remote.origin.fetch 2>/dev/null || echo '(unset)')"
else
    echo "git repo    : NO  (.git is absent)"
fi

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

# If already matching requested pin and >= MIN_VERSION, skip git network fetch
if [ -n "$PIN_REF" ] && [ "$BAKED" = "${PIN_REF#v}" ] && ver_ge "$BAKED" "$MIN_VERSION"; then
    echo "RESULT: baked ComfyUI ${BAKED} matches ${PIN_REF} and satisfies >= ${MIN_VERSION}."
elif [ -z "$PIN_REF" ]; then
    if ver_ge "$BAKED" "$MIN_VERSION"; then
        echo "RESULT: baked ComfyUI ${BAKED} satisfies >= ${MIN_VERSION}."
    else
        echo "RESULT: FAIL - baked ComfyUI ${BAKED} is below the required ${MIN_VERSION}."
        exit 1
    fi
else
    echo "Explicit checkout to '${PIN_REF}'..."
    DEPTH_ARG=""
    if [ -f .git/shallow ]; then DEPTH_ARG="--depth 1"; fi
    
    # Try fetching tag from origin or upstream Comfy-Org
    git fetch ${DEPTH_ARG} origin "+refs/tags/${PIN_REF}:refs/tags/${PIN_REF}" 2>/dev/null \
        || git fetch ${DEPTH_ARG} https://github.com/Comfy-Org/ComfyUI.git "+refs/tags/${PIN_REF}:refs/tags/${PIN_REF}" 2>/dev/null \
        || git fetch --tags --force origin 2>/dev/null \
        || true

    git checkout --detach "refs/tags/${PIN_REF}" 2>/dev/null \
        || git checkout --detach "${PIN_REF}" 2>/dev/null \
        || echo "WARN: checkout of ${PIN_REF} had non-zero exit; verifying version file directly."
fi

# --------------------------------------------------- realign pinned helper pkgs
if [ -f requirements.txt ]; then
    echo "realigning helper packages to this revision's pins..."
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"
    python3 /usr/local/bin/filter-req.py requirements.txt /tmp/comfy-req.filtered
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 -r /tmp/comfy-req.filtered || echo "WARN: requirements install returned non-zero (non-fatal)"
    TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
        echo "FATAL: ComfyUI requirements.txt moved torch ${TORCH_BEFORE} -> ${TORCH_AFTER}"
        exit 1
    fi
fi

FINAL_VERSION="$(read_version)"
echo "final ComfyUI version: ${FINAL_VERSION}"

if ! ver_ge "$FINAL_VERSION" "$MIN_VERSION"; then
    echo
    echo "FATAL: ComfyUI ${FINAL_VERSION} is still below the required minimum ${MIN_VERSION}."
    exit 1
fi
echo "gate        : ${FINAL_VERSION} >= ${MIN_VERSION} OK"
echo "=============================================================="
