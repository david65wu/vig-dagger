#!/usr/bin/env bash
# Install the Isaac Sim environment OpenDrawer's simulator jobs run in (demo collection, DAgger, evaluation):
#   <repo>/.venv-isaac               Python 3.11 with Isaac Sim 5.1, PyTorch 2.7.0 (CUDA 12.8) and Isaac Lab v2.3.2
#   <repo>/third_party/IsaacLab      Isaac Lab's source at tag v2.3.2 (installed editable)
# Training, conversion and analysis use the main environment (`uv sync`), which has a newer PyTorch.
#
# Needs: Linux x86_64 with glibc >= 2.35 (Ubuntu 22.04 or newer), an NVIDIA GPU and driver that Isaac Sim 5.1
# supports, uv, git, internet access, about 30 GB of disk. Takes 10-30 minutes, mostly downloads.
# Rerunning it is safe: what is installed already is kept.
#
#   bash scripts/install_isaac.sh                 # then: vig-dagger check -c configs/opendrawer.yaml
# Optional: ISAAC_ENV=<dir> and ISAACLAB_DIR=<dir> put the environment / the Isaac Lab source elsewhere.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ISAAC_ENV:-$REPO/.venv-isaac}"
LAB_DIR="${ISAACLAB_DIR:-$REPO/third_party/IsaacLab}"
LAB_TAG="v2.3.2"  # the Isaac Lab release for Isaac Sim 5.1 (Isaac Lab 0.54.2)
PY="$ENV_DIR/bin/python"
NVIDIA_INDEX=(--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match)
OVERRIDES="$ENV_DIR/vig-dagger-overrides.txt"

step() { printf '\n==== %s\n' "$*"; }
command -v uv >/dev/null || { echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v git >/dev/null || { echo "git is not installed"; exit 1; }

step "1/5 Python 3.11 environment: $ENV_DIR"
[ -x "$PY" ] || uv venv --python 3.11 "$ENV_DIR"
# Keep Isaac Sim's PyTorch whatever later packages ask for; Isaac Lab v2.3.2 pins an old flatdict that no longer builds.
printf 'torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\nflatdict>=4.1.0\n' > "$OVERRIDES"

step "2/5 PyTorch 2.7.0 for CUDA 12.8 (the version Isaac Sim 5.1 is built against)"
uv pip install --python "$PY" "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" --default-index https://download.pytorch.org/whl/cu128

step "3/5 Isaac Sim 5.1 (NVIDIA's pip packages, about 10 GB)"
uv pip install --python "$PY" "isaacsim[all,extscache]==5.1.0" "${NVIDIA_INDEX[@]}" --override "$OVERRIDES"

step "4/5 Isaac Lab $LAB_TAG"
if [ ! -d "$LAB_DIR/.git" ]; then
  mkdir -p "$(dirname "$LAB_DIR")"
  git clone --depth 1 --branch "$LAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$LAB_DIR"
fi
uv pip install --python "$PY" -e "$LAB_DIR/source/isaaclab" -e "$LAB_DIR/source/isaaclab_assets" \
  "${NVIDIA_INDEX[@]}" --override "$OVERRIDES"

step "5/5 this repository (vig_dagger and the robopolicy library) and what its simulator jobs import"
uv pip install --python "$PY" --no-deps -e "$REPO"
uv pip install --python "$PY" "pydantic>=2.11,<3" "transformers>=4.56,<5" "typing-extensions>=4.12" h5py psutil pyyaml \
  --override "$OVERRIDES"

step "check"
OMNI_KIT_ACCEPT_EULA=YES "$PY" -c "
import importlib.metadata as m, torch
import isaaclab, vig_dagger, robopolicy  # noqa: F401
assert torch.__version__.startswith('2.7.0+cu128') and torch.version.cuda == '12.8', torch.__version__
print('isaacsim', m.version('isaacsim'), '| isaaclab', m.version('isaaclab'), '| torch', torch.__version__,
      '| CUDA', torch.cuda.is_available())
"
echo "Installed. Next: vig-dagger check -c configs/opendrawer.yaml (it also starts Isaac Sim once to test rendering)."
