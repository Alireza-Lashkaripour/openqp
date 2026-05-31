#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Builds the native liboqp library and installs the Python dependencies so that
# `import oqp` and the pytest suite work from the source tree. Runs
# synchronously: the session waits until the environment is ready, which avoids
# the agent trying to build/test before dependencies exist.
set -euo pipefail

# Only run in the remote (web) environment; local checkouts manage their own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"

BUILD_LOG="/tmp/oqp_session_start_build.log"

log() { echo "[session-start] $*" >&2; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

log "installing system build dependencies (gfortran, OpenBLAS, LAPACK, cmake, ninja)..."
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq || true
$SUDO apt-get install -y -qq \
  gfortran libopenblas-dev liblapack-dev cmake ninja-build build-essential >/dev/null

log "installing Python dependencies..."
python3 -m pip install -q numpy scipy cffi pyscf basis_set_exchange libdlfind pytest
# geomeTRIC's setup.py trips Debian's patched setuptools (install_layout); force
# the stdlib distutils so the source build succeeds.
SETUPTOOLS_USE_DISTUTILS=stdlib python3 -m pip install -q "geometric>=1.0"

# Build the native library. Idempotent: skip if it already exists on a warm
# container (session resume / clear / compact).
if [ ! -f "$REPO/lib/liboqp.so" ]; then
  log "building liboqp with Rys-quadrature ERIs (USE_LIBINT=OFF); this can take several minutes..."
  if ! cmake -B "$REPO/build" -G Ninja \
      -DUSE_LIBINT=OFF \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
      -DCMAKE_INSTALL_PREFIX=. -DENABLE_OPENMP=ON -DLINALG_LIB_INT64=OFF \
      >"$BUILD_LOG" 2>&1; then
    log "cmake configure failed; last lines of $BUILD_LOG:"; tail -40 "$BUILD_LOG" >&2; exit 1
  fi
  if ! ninja -C "$REPO/build" install >>"$BUILD_LOG" 2>&1; then
    log "ninja build failed; last lines of $BUILD_LOG:"; tail -40 "$BUILD_LOG" >&2; exit 1
  fi
  log "liboqp build complete."
else
  log "liboqp already built; skipping native build."
fi

# Persist environment for the session so the tests find the native library and
# the source package (we run from the tree rather than pip-installing the wheel).
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export OPENQP_ROOT=$REPO"
    echo "export PYTHONPATH=$REPO/pyoqp"
    echo "export OMP_NUM_THREADS=2"
  } >> "$CLAUDE_ENV_FILE"
fi

log "setup complete; run the suite with: python3 -m pytest tests/"
