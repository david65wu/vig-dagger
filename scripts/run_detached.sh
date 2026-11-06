#!/usr/bin/env bash
# Run in a new session with stdin /dev/null and stdout/stderr redirected by the caller.
# Strict gates: setup -> tests -> simulator preflight -> end-to-end smoke -> experiment.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP=/cpfs02/user/caojianuo/vig-dagger-runs/setup
cd "$REPO"
exec 9>"$SETUP/startup.lock"
flock -n 9 || { echo 'Another startup process is active'; exit 1; }
trap 'code=$?; if (( code != 0 )); then echo "Startup stopped with exit $code at $(date -Is); inspect startup.log and stage logs"; fi' EXIT
export PATH="$REPO/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export UV_CACHE_DIR=/cpfs02/user/caojianuo/.cache/uv
export HF_HOME=/cpfs02/user/caojianuo/.cache/huggingface
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONUNBUFFERED=1
while [[ ! -f "$SETUP/INSTALL_DONE" ]]; do
  kill -0 "$(cat "$SETUP/install.pid")" 2>/dev/null || { echo 'Installation did not complete; inspect install.log'; exit 1; }
  sleep 10
done
source "$SETUP/graphics-env.sh"
echo "Tests: $(date -Is)"
python -m pytest tests/test_takeover_sweep.py tests/test_failed_retention.py -q > "$SETUP/tests.log" 2>&1
if [[ ! -f "$SETUP/PREFLIGHT_DONE" ]]; then
  echo "Preflight: $(date -Is)"
  vig-dagger check -c configs/run.yaml --set 'resources.gpus=[7]' > "$SETUP/preflight.log" 2>&1
  touch "$SETUP/PREFLIGHT_DONE"
fi
echo "Smoke: $(date -Is)"
vig-dagger run -c configs/smoke_sweep.yaml --set work_dir=/cpfs02/user/caojianuo/vig-dagger-runs/smoke > "$SETUP/smoke.log" 2>&1
touch "$SETUP/SMOKE_DONE"
while [[ ! -f "$SETUP/DEMOS_DONE" ]]; do
  kill -0 "$(cat "$SETUP/download-demos.pid")" 2>/dev/null || { echo 'Demo download did not complete; inspect download-demos.log'; exit 1; }
  sleep 10
done
while [[ ! -f "$SETUP/PREP_DONE" ]]; do
  kill -0 "$(cat "$SETUP/prepare-data.pid")" 2>/dev/null || { echo 'Data preparation did not complete; inspect prepare-data.log'; exit 1; }
  sleep 10
done
echo "Experiment: $(date -Is)"
exec vig-dagger run -c configs/run.yaml >> /cpfs02/user/caojianuo/vig-dagger-runs/opendrawer_takeover_sweep_v1/run.log 2>&1
