#!/usr/bin/env bash
#
# Writes a build manifest describing exactly what landed in the image.
#
# This is a DIAGNOSTIC artifact. It must never fail the build, so every command
# is individually guarded and the script always exits 0. Do not add `set -e`.
#
# Usage: build-manifest.sh [output-path]
#   COMFY_DIR  override the ComfyUI install path (default /opt/ComfyUI)

set -u

COMFY="${COMFY_DIR:-/opt/ComfyUI}"
OUT="${1:-/opt/build-manifest.txt}"

# git refuses to read repositories owned by a different uid. Harmless here.
git config --global --add safe.directory '*' >/dev/null 2>&1 || true

# Both quote characters, built without ever nesting a quote inside its own kind.
QUOTES="\"'"

# Run a python snippet, printing "unknown" instead of failing.
py() {
	python3 -c "$1" 2>/dev/null || echo "unknown"
}

comfy_version() {
	f="$COMFY/comfyui_version.py"
	if [ ! -f "$f" ]; then
		echo "missing"
		return 0
	fi
	v="$(grep -m1 '__version__' "$f" 2>/dev/null | cut -d= -f2- | tr -d " \r${QUOTES}")"
	if [ -n "$v" ]; then
		echo "$v"
	else
		echo "unparsed"
	fi
}

{
	echo "# Build manifest"
	echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
	echo "torch: $(py 'import torch;print(torch.__version__)')"
	echo "torch_cuda: $(py 'import torch;print(torch.version.cuda)')"
	echo "sage_arch: $(cat /etc/sage-arch 2>/dev/null || echo unknown)"
	echo "sageattention: $(py 'import sageattention as s;print(getattr(s, "__version__", "present"))')"
	echo "comfyui_version: $(comfy_version)"
	echo "comfyui_git: $(git -C "$COMFY" describe --tags --always 2>/dev/null || echo unknown)"
	echo "custom_nodes:"
	for d in "$COMFY"/custom_nodes/*/; do
		[ -d "$d" ] || continue
		name="$(basename "$d")"
		if [ -d "$d/.git" ]; then
			echo "  ${name}: $(git -C "$d" rev-parse HEAD 2>/dev/null || echo unknown)"
		else
			# COPY'd packs such as comfyui-h3-mlp-chunk have no .git. List them
			# anyway: their presence is exactly what we need to verify.
			echo "  ${name}: (no git, copied into image)"
		fi
	done
} >"$OUT" 2>/dev/null

cat "$OUT" 2>/dev/null || echo "build-manifest: could not write $OUT"
exit 0
