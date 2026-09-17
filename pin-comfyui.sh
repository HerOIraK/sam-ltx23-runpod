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
MIN_VERSION="${COMFYUI_MIN_VERSION:-0.36.0}"
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

# ------------------------------------------------------------ verify-only path
if [ -z "$PIN_REF" ]; then
    if ver_ge "$BAKED" "$MIN_VERSION"; then
        echo "RESULT: baked ComfyUI ${BAKED} satisfies >= ${MIN_VERSION}."
        echo "        Leaving the checkout untouched."
        exit 0
    fi
    echo "RESULT: FAIL - baked ComfyUI ${BAKED} is below the required ${MIN_VERSION}."
    exit 1
fi

# ------------------------------------------------------------- explicit pin
echo "Explicit pin requested. Checking that '${PIN_REF}' exists on origin..."

is_tag=0
is_branch=0
if git ls-remote --tags --refs origin 2>/dev/null | grep -q "refs/tags/${PIN_REF}\$"; then
    echo "  found refs/tags/${PIN_REF}"
    is_tag=1
elif git ls-remote --heads origin 2>/dev/null | grep -q "refs/heads/${PIN_REF}\$"; then
    echo "  found refs/heads/${PIN_REF} (branch, not tag)"
    is_branch=1
else
    echo "  Attempting upstream fetch..."
fi

fetched=0

if [ -f .git/shallow ]; then
    DEPTH_ARG="--depth 1"
    echo "repo is shallow      -> fetching with --depth 1"
else
    DEPTH_ARG=""
    echo "repo is a full clone -> fetching without --depth (history preserved)"
fi

if [ "$is_tag" -eq 1 ]; then
    echo "attempt tag fetch: refs/tags/${PIN_REF}"
    if git fetch ${DEPTH_ARG} origin "+refs/tags/${PIN_REF}:refs/tags/${PIN_REF}" 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$is_branch" -eq 1 ] && [ "$fetched" -eq 0 ]; then
    echo "attempt branch fetch: origin/${PIN_REF}"
    if git fetch ${DEPTH_ARG} origin "+refs/heads/${PIN_REF}:refs/remotes/origin/${PIN_REF}" 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt: commit or ref fetch"
    if git fetch ${DEPTH_ARG} origin "${PIN_REF}" 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt: upstream tag refspec fallback"
    if git fetch ${DEPTH_ARG} https://github.com/Comfy-Org/ComfyUI.git "+refs/tags/${PIN_REF}:refs/tags/${PIN_REF}" 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt: upstream commit/branch fetch fallback"
    if git fetch ${DEPTH_ARG} https://github.com/Comfy-Org/ComfyUI.git "${PIN_REF}" 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt: full tag fetch"
    if git fetch --tags --force origin 2>/dev/null; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt: unshallow, then full tag fetch"
    git fetch --unshallow origin 2>/dev/null || true
    if git fetch --tags --force origin 2>/dev/null; then
        fetched=1
    fi
fi

if git checkout --detach "refs/tags/${PIN_REF}" 2>/dev/null \
    || git checkout --detach "origin/${PIN_REF}" 2>/dev/null \
    || git checkout --detach "${PIN_REF}" 2>/dev/null \
    || git checkout --detach FETCH_HEAD 2>/dev/null; then
    echo "checked out: $(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
    echo "FATAL: failed to checkout '${PIN_REF}' after fetch."
    exit 1
fi

# --------------------------------------------------- realign pinned helper pkgs
if [ -f requirements.txt ]; then
    echo "realigning helper packages to this revision's pins..."
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"
    python3 /usr/local/bin/filter-req.py requirements.txt /tmp/comfy-req.filtered
    pip install --no-cache-dir --no-deps -r /tmp/comfy-req.filtered || true
    TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
        echo "FATAL: ComfyUI requirements.txt moved torch ${TORCH_BEFORE} -> ${TORCH_AFTER}"
        exit 1
    fi
fi

FINAL_VERSION="$(read_version)"
echo "final ComfyUI version: ${FINAL_VERSION}"

if ! ver_ge "$FINAL_VERSION" "$MIN_VERSION"; then
    echo "FATAL: pinned ref '${PIN_REF}' resolves to ComfyUI ${FINAL_VERSION}, below minimum ${MIN_VERSION}."
    exit 1
fi
echo "gate        : ${FINAL_VERSION} >= ${MIN_VERSION} OK"
echo "=============================================================="
