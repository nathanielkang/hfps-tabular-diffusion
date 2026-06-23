#!/usr/bin/env bash
# setup_env.sh - one-time environment setup on the VM.
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv tmux
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
# CPU-only torch (no CUDA) + data stack
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pyarrow scikit-learn scipy tqdm gdown psutil
echo "env ready. activate with: source ~/venv/bin/activate"