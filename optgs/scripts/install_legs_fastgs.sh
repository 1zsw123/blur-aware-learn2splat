#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEGS_ROOT="$ROOT/third_party/LeGS"
EXTENSION_ROOT="$LEGS_ROOT/submodules/diff-gaussian-rasterization_fastgs"
PATCH_FILE="$ROOT/third_party/patches/legs-fastgs-cuda12-cstdint.patch"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXPECTED_LEGS_COMMIT="8eb120b1f0c0fe0727e0440f4e372b412f275572"
ACTUAL_LEGS_COMMIT="$(git -C "$LEGS_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_LEGS_COMMIT" != "$EXPECTED_LEGS_COMMIT" ]]; then
  echo "LeGS commit mismatch: expected $EXPECTED_LEGS_COMMIT, got $ACTUAL_LEGS_COMMIT" >&2
  exit 2
fi

BUILD_PARENT="${BUILD_PARENT:-${TMPDIR:-/tmp}}"
BUILD_ROOT="$(mktemp -d "$BUILD_PARENT/learn2splat-legs-fastgs.XXXXXX")"
cleanup() {
  rm -rf "$BUILD_ROOT"
}
trap cleanup EXIT

cp -a "$EXTENSION_ROOT/." "$BUILD_ROOT/source/"
patch -d "$BUILD_ROOT/source" -p1 < "$PATCH_FILE"
mkdir -p "$BUILD_ROOT/tmp"

TMPDIR="$BUILD_ROOT/tmp" "$PYTHON_BIN" -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "$BUILD_ROOT/source"

"$PYTHON_BIN" -c \
  'import diff_gaussian_rasterization_fastgs as m; print(m.__file__)'
