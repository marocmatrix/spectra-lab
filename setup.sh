#!/usr/bin/env bash
# setup.sh — install Spectra-Lab into a venv on a Raspberry Pi.
# Run once from inside the app folder:  bash setup.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing Spectra-Lab in: $APP_DIR"

# System build deps for scientific Python wheels on ARM (safe if already present)
if command -v apt-get >/dev/null 2>&1; then
  echo "Ensuring build/runtime libraries are present (may prompt for sudo)…"
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-dev build-essential \
       libatlas-base-dev libopenblas-dev gfortran
fi

# Create / refresh the virtual environment
python3 -m venv "$APP_DIR/.venv"
# shellcheck disable=SC1091
source "$APP_DIR/.venv/bin/activate"

pip install --upgrade pip wheel
echo "Installing Python dependencies (this can take a few minutes on a Pi)…"
pip install -r "$APP_DIR/requirements.txt"

echo
echo "Done. Test it with:"
echo "  source $APP_DIR/.venv/bin/activate && streamlit run $APP_DIR/app.py"
