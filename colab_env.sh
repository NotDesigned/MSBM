export PIP_CACHE_DIR=/content/drive/MyDrive/.pip-cache
mkdir -p "$PIP_CACHE_DIR"
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio --upgrade
pip install -r requirements_colab.txt