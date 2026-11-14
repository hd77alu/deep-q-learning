#!/usr/bin/env bash
set -euo pipefail
MEMBER="${MEMBER:-mahamat_bakhit}"
RUN="${1:-all}"
run_one() {
  local id="$1" lr=0.0001 gamma=0.99 batch=64 start=1.0 end=0.05 decay=50000
  case "$id" in
    1) ;;
    2) lr=0.00005 ;;
    3) lr=0.0002 ;;
    4) gamma=0.95 ;;
    5) gamma=0.995 ;;
    6) batch=32 ;;
    7) batch=128 ;;
    8) start=0.8 ;;
    9) end=0.02 ;;
    10) decay=25000 ;;
    *) echo "Experiment must be 1-10" >&2; exit 2 ;;
  esac
  if [[ -f "$MEMBER/exp_$(printf "%02d" "$id")/completed.json" ]]; then
    echo "Experiment $id already completed; skipping"
    return
  fi
  uv run python train.py --member-name "$MEMBER" --exp-id "$id" --lr "$lr" --gamma "$gamma" --batch-size "$batch" --eps-start "$start" --eps-end "$end" --eps-decay "$decay" --total-steps 100000 --buffer-capacity 20000
}
if [[ "$RUN" == all ]]; then
  for id in {1..10}; do run_one "$id"; done
else
  run_one "$RUN"
fi
