#!/usr/bin/env bash
# Create a reproducible UniInfer environment without root, conda or system venv support.
#
#   UNIINFER_WORK=/path/with/space bash scripts/setup_env.sh
#   source "$UNIINFER_WORK/env.sh"
#
# UNIINFER_WORK holds the venv, uv, the Python toolchain and the Hugging Face cache (models are about 8 GB).
# Put it on a large scratch disk, not a quota-limited home directory.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${UNIINFER_WORK:-$REPO/.work}"
mkdir -p "$WORK"/{tools,hf_cache,logs}
chmod 700 "$WORK"
export UV_CACHE_DIR="$WORK/hf_cache/uv" UV_PYTHON_INSTALL_DIR="$WORK/python"

python3 -m pip install -q --target "$WORK/tools" uv==0.5.11
export PYTHONPATH="$WORK/tools"
UV="$WORK/tools/bin/uv"

[ -d "$WORK/venv" ] || "$UV" venv --python 3.11 "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
"$UV" pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$UV" pip install -e "$REPO[experiments,dev]"

cat > "$WORK/env.sh" <<EOT
export UNIINFER_WORK=$WORK
export HF_HOME=$WORK/hf_cache
export UV_CACHE_DIR=$WORK/hf_cache/uv
export UV_PYTHON_INSTALL_DIR=$WORK/python
export PYTHONPATH=$WORK/tools\${PYTHONPATH:+:\$PYTHONPATH}
export PATH=$WORK/tools/bin:\$PATH
export TOKENIZERS_PARALLELISM=false
source $WORK/venv/bin/activate
EOT

python -c "import torch, transformers, uniinfer; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| uniinfer', uniinfer.__version__)"
echo "Done. Next:  source $WORK/env.sh"
