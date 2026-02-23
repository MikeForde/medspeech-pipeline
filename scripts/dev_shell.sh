# Usage (zsh or bash):
#   source scripts/dev_shell.sh
# or
#   . scripts/dev_shell.sh

# Don't use "set -u" in a sourced script; it breaks on unset prompt vars in VS Code zsh.
set -e

ENV_NAME="medspeech-audio"
BIN_DIR="$HOME/bin"
WRAPPER="$BIN_DIR/demucs"

# Resolve repo root in a shell-agnostic way.
# In zsh, ${(%):-%N} gives the current script path; in bash, ${BASH_SOURCE[0]} works.
if [ -n "${ZSH_VERSION:-}" ]; then
  SCRIPT_PATH="${(%):-%N}"
else
  SCRIPT_PATH="${BASH_SOURCE:-$0}"
fi

REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

echo "== MedSpeech dev shell =="
echo "Repo: $REPO_ROOT"
echo "Shell: ${SHELL:-unknown}"
echo

# 1) Ensure ~/bin exists and is on PATH
mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"

# 2) Create/refresh demucs wrapper every time (hardcode micromamba path)
cat > "$WRAPPER" <<EOF
#!/usr/bin/env sh
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
exec /opt/homebrew/bin/micromamba run -n $ENV_NAME demucs "\$@"
EOF
chmod +x "$WRAPPER"

# 3) Activate Python .venv
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.venv/bin/activate"
else
  echo "ERROR: .venv not found at $REPO_ROOT/.venv"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt (or install deps)"
  return 2
fi

# 4) Verify micromamba + env
if [ ! -x "/opt/homebrew/bin/micromamba" ]; then
  echo "ERROR: micromamba not found at /opt/homebrew/bin/micromamba"
  echo "Fix: brew install micromamba"
  return 3
fi

if ! /opt/homebrew/bin/micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "WARNING: micromamba env '$ENV_NAME' not found."
  echo "Create it with:"
  echo "  micromamba create -n $ENV_NAME -c conda-forge python=3.11 -y"
  echo "  micromamba run -n $ENV_NAME python -m pip install --upgrade pip"
  echo "  micromamba run -n $ENV_NAME python -m pip install demucs torchcodec"
  return 4
fi

# 5) Verify demucs runnable via wrapper
if ! demucs -h >/dev/null 2>&1; then
  echo "ERROR: demucs wrapper exists but demucs failed to run."
  echo "Try:"
  echo "  /opt/homebrew/bin/micromamba run -n $ENV_NAME demucs -h"
  return 5
fi

echo "READY ✅"
echo "  Python: $(python --version 2>/dev/null || true)"
echo "  Venv:   ${VIRTUAL_ENV:-<not set>}"
echo "  demucs: $WRAPPER"
echo
echo "Try:"
echo "  python -m medspeech.cli samples/output.mp3"