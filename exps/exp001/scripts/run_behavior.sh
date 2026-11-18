#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == '--help' ]]; then
  echo 'Usage: bash run_behavior.sh [01|02|03|04|all|path/to/spec.yaml] [lanes=1]'
  echo '  lanes: parallel simulator lanes sharing one vLLM pair (1-4; ~26 GB GPU and ~5 CPU cores each)'
  exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/behavior_env.sh"
exec 9>"$EXP_DIR/logs/behavior_runtime.lock"
flock -n 9 || { echo '[error] BEHAVIOR setup or collection is already running' >&2; exit 1; }
LOG="$EXP_DIR/logs/collect_behavior-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "[run] log: $LOG"
[[ -f "$RUNTIME_DIR/.setup-complete" ]] || { echo '[error] Run setup_behavior.sh successfully first' >&2; exit 1; }
# pick_object grasps with GraspGen-X (EXP001_GRASP_PLANNER=graspgenx in the specs).
curl -sf -m 5 http://127.0.0.1:8390/health >/dev/null || {
  echo '[error] GraspGen-X server is not up on :8390 (see BEHAVIOR.md: src/tools/graspgenx_server.py)' >&2; exit 1; }
"$BASELINE_DIR/.venv/bin/python" "$EXP_DIR/src/behavior_collect/patch_runtime.py"
git -C "$RUNTIME_DIR/BEHAVIOR-1K" diff > "$RUNTIME_DIR/compatibility.patch"
selected="${1:-all}"
lanes="${2:-1}"
[[ "$lanes" =~ ^[1-4]$ ]] || { echo "[error] lanes must be 1-4, got $lanes" >&2; exit 1; }
case "$selected" in
  01|02|03|04) specs=("$EXP_DIR/configs/collect_behavior_part${selected}.yaml");;
  all) specs=("$EXP_DIR/configs/collect_behavior_part"{01,02,03,04}.yaml);;
  *) specs=("$(realpath -- "$selected")");;
esac
for spec in "${specs[@]}"; do
  # Resolve output paths with the same loader and guard used for planning.
  run_dir=$("$BASELINE_DIR/.venv/bin/python" -c \
    'from pathlib import Path; import sys; from behavior_collect.plan import validated_spec; print(validated_spec(Path(sys.argv[1])).output_dir)' "$spec")
  digest=$("$BASELINE_DIR/.venv/bin/python" -c \
    'from pathlib import Path; import sys; from behavior_collect.plan import collection_digest; print(collection_digest(Path(sys.argv[1])))' "$spec")
  if [[ -f "$run_dir/.collection-complete" && "$(< "$run_dir/.collection-complete")" == "$digest" ]]; then
    echo "[skip] $run_dir"
    continue
  fi
  [[ ! -e "$run_dir" ]] || { echo "[error] Existing output: $run_dir. Use a new spec/output for missing instances." >&2; exit 1; }
  "$BASELINE_DIR/.venv/bin/python" - "$spec" "$lanes" <<'PY'
import socket
import sys
from pathlib import Path
from behavior_collect.plan import sidecar_ports, validated_spec
from geniac_baseline.collect.plan import servers
spec = validated_spec(Path(sys.argv[1]))
ports = {s.port for s in servers(spec)[0]} | set(sidecar_ports(spec, int(sys.argv[2])))
for port in sorted(ports):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is occupied; stop the other collection or use a new spec with port offsets.')
PY
  "$BASELINE_DIR/.venv/bin/python" "$EXP_DIR/src/behavior_collect/plan.py" --spec "$spec" --lanes "$lanes"
  time bash "$run_dir/run.sh"
  "$BASELINE_DIR/.venv/bin/python" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
from geniac_baseline.eval.score import score_runs
r = Path(sys.argv[1])
score = score_runs([r])
infra = list(r.glob('seg_*/outputs/*/*/trial_*/infrastructure_failure.json'))
score['infrastructure_failures'] = [str(p) for p in infra]
(r / 'score.json').write_text(json.dumps(score, indent=2))
print(score['total'])
if infra:
    print(f'[warning] {len(infra)} infrastructure failures recorded separately; excluded from training.')
if score['total']['missing']:
    raise SystemExit('Missing final trial artifacts; collection is incomplete. See score.json.')
PY
  if [[ "$("$BASELINE_DIR/.venv/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("infrastructure_failures", [])))' "$run_dir/score.json")" == 0 ]]; then
    printf '%s\n' "$digest" > "$run_dir/.collection-complete"
  else
    printf '%s\n' "$digest" > "$run_dir/.collection-attempted"
  fi
done
