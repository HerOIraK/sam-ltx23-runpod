#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ComfyUI version gate.
#
# Reports the current state BEFORE changing anything, then:
#   - COMFYUI_REF empty  -> verify only. Pass if baked version >= minimum.
#                           Never mutates the repo. This is the default.
#   - COMFYUI_REF set    -> move the checkout to that ref, with correct
#                           shallow-tag refspecs and escalating fallbacks.
#
# Rationale: the base image usually already ships a new enough ComfyUI. Moving
# a working checkout is riskier than leaving it alone, so we only do it when
# explicitly told to.
# ---------------------------------------------------------------------------
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
MIN_VERSION="${COMFYUI_MIN_VERSION:-0.31.0}"
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
    echo "              A git-based pin is impossible. Rely on the version file."
fi

read_version() {
    python3 - <<'PY'
import re, pathlib, sys
# 1. comfyui_version.py is generated at release time and is authoritative.
p = pathlib.Path("comfyui_version.py")
if p.exists():
    m = re.search(r"""__version__\s*=\s*['\"]([^'\"]+)['\"]""", p.read_text())
    if m:
        print(m.group(1)); sys.exit(0)
# 2. fall back to pyproject.toml
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
    echo
    echo "The MiniMax H3 core nodes this workflow uses require ${MIN_VERSION}+."
    echo "Either use a newer base image, or rebuild with:"
    echo "    --build-arg COMFYUI_REF=v${MIN_VERSION}"
    echo
    echo "Tags currently on origin (last 20):"
    git ls-remote --tags --refs origin 2>/dev/null \
        | awk -F/ '{print "    " $NF}' | tail -n 20 \
        || echo "    (could not reach origin)"
    exit 1
fi

# ------------------------------------------------------------- explicit pin
echo "Explicit pin requested. Checking that '${PIN_REF}' exists on origin..."

if git ls-remote --tags --refs origin 2>/dev/null | grep -q "refs/tags/${PIN_REF}\$"; then
    echo "  found refs/tags/${PIN_REF}"
elif git ls-remote --heads origin 2>/dev/null | grep -q "refs/heads/${PIN_REF}\$"; then
    echo "  found refs/heads/${PIN_REF} (branch, not tag)"
else
    echo "  NOT FOUND on origin. This is why a plain 'git fetch origin <ref>' fails"
    echo "  with: fatal: couldn't find remote ref ${PIN_REF}"
    echo
    echo "  origin URL: $(git remote get-url origin 2>/dev/null || echo unknown)"
    echo "  If that URL is not the upstream ComfyUI repository, the tag will never"
    echo "  resolve no matter how the refspec is written. Add the upstream as a"
    echo "  second remote instead:"
    echo "      git remote add upstream https://github.com/Comfy-Org/ComfyUI.git"
    echo
    echo "  origin tags (last 20):"
    git ls-remote --tags --refs origin 2>/dev/null | awk -F/ '{print "    " $NF}' | tail -n 20 || true
    echo "  origin heads (last 20):"
    git ls-remote --heads origin 2>/dev/null | awk -F/ '{print "    " $NF}' | tail -n 20 || true
    exit 1
fi

fetched=0

# Correct shallow tag refspec. A bare 'git fetch --depth 1 origin v0.31.0' can
# fail on repos cloned with a single-branch fetch refspec; naming both sides
# explicitly avoids that.
echo "attempt 1: shallow explicit tag refspec"
if git fetch --depth 1 origin "+refs/tags/${PIN_REF}:refs/tags/${PIN_REF}"; then
    fetched=1
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt 2: shallow explicit branch refspec"
    if git fetch --depth 1 origin "+refs/heads/${PIN_REF}:refs/remotes/origin/${PIN_REF}"; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt 3: full tag fetch"
    if git fetch --tags --force origin; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "attempt 4: unshallow, then full tag fetch"
    git fetch --unshallow origin 2>/dev/null || true
    if git fetch --tags --force origin; then
        fetched=1
    fi
fi

if [ "$fetched" -eq 0 ]; then
    echo "FATAL: could not fetch '${PIN_REF}' from origin after 4 attempts."
    exit 1
fi

git checkout --detach "refs/tags/${PIN_REF}" 2>/dev/null \
    || git checkout --detach "${PIN_REF}" \
    || { echo "FATAL: fetched '${PIN_REF}' but could not check it out."; exit 1; }

echo "checked out: $(git describe --tags --always)"

# --------------------------------------------------- realign pinned helper pkgs
if [ -f requirements.txt ]; then
    echo "realigning helper packages to this revision's pins..."
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"
    python3 /usr/local/bin/filter-req.py requirements.txt /tmp/comfy-req.filtered
    pip install --no-cache-dir -r /tmp/comfy-req.filtered
    TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
        echo "FATAL: ComfyUI requirements.txt moved torch ${TORCH_BEFORE} -> ${TORCH_AFTER}"
        exit 1
    fi
fi

echo "final ComfyUI version: $(read_version)"
echo "=============================================================="
