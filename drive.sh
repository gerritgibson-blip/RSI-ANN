#!/bin/bash
# Keep resuming until the run stops for a reason other than the time budget.
# Training runs inside a memory cgroup with a hard 11 GB cap. If the cap kills it (exit 137),
# the memory-efficiency ladder is raised and training resumes; at the top of the ladder the
# run stops with hardware_memory_limit.
cd "$(dirname "$0")"
CG=/sys/fs/cgroup/memory/selfimprove
CAP=$((11*1024*1024*1024))
setup_cg() {
  mkdir -p $CG
  echo $CAP > $CG/memory.limit_in_bytes
  echo $CAP > $CG/memory.memsw.limit_in_bytes 2>/dev/null || true
  echo $$ > $CG/cgroup.procs   # this shell and its children are now capped
}
setup_cg
while pgrep -f "python3 -u selfimprove.py" >/dev/null; do sleep 10; done
while true; do
  python3 -u selfimprove.py --resume --budget_min 110 --round_steps 400 >> run.out 2>&1
  rc=$?
  if [ $rc -eq 137 ]; then
    lvl=$(python3 - <<'PY'
import json, selfimprove as S
c = json.load(open("counters.json")); l = c.get("mem_level", 0)
if c.get("mem_ladder", 1) != S.MEM_LADDER_VERSION:
    l = S.OLD_LADDER_MAP.get(l, len(S.MEM_LADDER) - 1)
import torch
nl = torch.load("ckpt.pt", map_location="cpu", weights_only=False)["cfg"]["n_layer"]
c.setdefault("level_peaks", {})[str(l)] = [S.MEM_CAP_GB, nl]  # this level does not fit at this depth
if l + 1 < len(S.MEM_LADDER):
    c["mem_level"] = l + 1; c["mem_ladder"] = S.MEM_LADDER_VERSION
    json.dump(c, open("counters.json", "w")); print(l + 1)
else:
    print("max")
PY
)
    if [ "$lvl" = "max" ]; then echo "$(date) chunk done: hardware_memory_limit (OOM at 11 GB cap, ladder exhausted)" >> drive.log; break; fi
    echo "$(date) OOM at 11 GB cap -> raised memory-efficiency level to $lvl, resuming" >> drive.log
    continue
  fi
  if [ $rc -ne 0 ]; then echo "crash rc=$rc $(date)" >> drive.log; sleep 30; continue; fi
  reason=$(tail -1 log.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.read()).get('stop_reason',''))")
  echo "$(date) chunk done: $reason" >> drive.log
  [ "$reason" != "time_budget" ] && break
done
echo "DONE $(date)" >> drive.log
