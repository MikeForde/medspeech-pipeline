#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative to where this script lives
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="medspeech-audio"
BIN_DIR="${HOME}/bin"
WRAPPER="${BIN_DIR}/demucs"

echo "== MedSpeech bootstrap (audio env) =="
echo "Repo: ${REPO_ROOT}"
echo "Env:  ${ENV_NAME}"
echo

# --- checks ---
if ! command -v micromamba >/dev/null 2>&1; then
  echo "ERROR: micromamba not found on PATH."
  echo "Fix: install micromamba (brew install micromamba) or ensure it is on PATH."
  exit 1
fi

mkdir -p "${BIN_DIR}"

# --- create wrapper ---
cat > "${WRAPPER}" <<'EOF'
#!/usr/bin/env bash
# Demucs wrapper to ensure it runs from the micromamba env and avoids broken HF tokens.
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
exec micromamba run -n medspeech-audio demucs "$@"
EOF

chmod +x "${WRAPPER}"

echo "Wrote wrapper: ${WRAPPER}"
echo

# --- ensure ~/bin is on PATH (for this shell session) ---
export PATH="${BIN_DIR}:${PATH}"

echo "PATH now begins with: ${BIN_DIR}"
echo

# --- check env exists ---
if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "WARNING: micromamba env '${ENV_NAME}' not found."
  echo "You can create it with:"
  echo "  micromamba create -n ${ENV_NAME} -c conda-forge python=3.11 -y"
  echo "  micromamba run -n ${ENV_NAME} python -m pip install --upgrade pip"
  echo "  micromamba run -n ${ENV_NAME} python -m pip install demucs torchcodec"
  echo
else
  echo "Found micromamba env: ${ENV_NAME}"
fi

echo
echo "Diagnostics:"
echo "  micromamba: $(command -v micromamba)"
echo "  demucs wrapper: $(command -v demucs || true)"
echo

echo "Testing: demucs -h (via wrapper)"
if demucs -h >/dev/null 2>&1; then
  echo "  OK: demucs is runnable"
else
  echo "  ERROR: demucs failed to run."
  echo "  Try:"
  echo "    micromamba run -n ${ENV_NAME} demucs -h"
  exit 2
fi

echo
echo "Done."
echo
echo "Tip (VS Code PATH issue on macOS):"
echo "If VS Code still can't find demucs after restart, run this script from the VS Code terminal,"
echo "or launch VS Code from a shell using: code ."
