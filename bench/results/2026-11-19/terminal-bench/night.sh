#!/usr/bin/env zsh
# Night run: finish Maya (left + 3 reruns), then Basalt (all 20), then restore Qwen.
# Both engines through llm-serve on :8000: deepseek-maya (Maya) and deepseek (Basalt, the adopted profile).
# Usage: TB_DIR=<dir> night.sh HH:MM   (start time today/tonight)
# TB_DIR holds tasks.txt (a copy of ../tb-tasks.txt) and summ.py; harbor jobs go to $TB_DIR/jobs.
set -u
T=${TB_DIR:?set TB_DIR to a work directory holding tasks.txt and summ.py}
log() { echo "$(date +%T) $*"; }

start=$(date -d "$1" +%s); now=$(date +%s)
(( start < now )) && start=$(( start + 86400 ))
log "waiting until $(date -d @$start)"
while (( $(date +%s) < start )); do sleep 30; done

export COLUMNS=200 OPENAI_API_KEY=local
MI='model_info={"max_input_tokens":262144,"max_output_tokens":32768,"input_cost_per_token":0,"output_cost_per_token":0}'
hb() {  # hb JOB PORT TASKFILE
  local I=(); while read t; do [[ -n $t ]] && I+=(-i "$t"); done < $3
  log "harbor $1: $(( ${#I[@]} / 2 )) tasks on :$2"
  (cd $T && uvx --from harbor harbor run -d terminal-bench@2.0 "${I[@]}" -a terminus-2 -m openai/deepseek-v4.1-flash \
     --ak api_base=http://127.0.0.1:$2/v1 --ak "$MI" -n 1 --agent-timeout-multiplier 2 \
     -o $T/jobs --job-name $1 -y -q > $T/$1.log 2>&1)
  log "harbor $1 rc=$?"
  python3 $T/summ.py $T/jobs/$1
}
wait_up() { for i in $(seq 1 240); do curl -sf -m 3 http://127.0.0.1:$1/v1/models >/dev/null && return 0; sleep 5; done; return 1; }

# Maya part 2: tasks not yet finished in tb20-maya, plus the three contended ones
python3 - $T <<'EOF' > $T/tasks-maya-b.txt
import json,glob,sys
T=sys.argv[1]; done=set()
for f in glob.glob(f'{T}/jobs/tb20-maya/*/result.json'):
    r=json.load(open(f))
    if (r.get('verifier_result') or {}).get('rewards'): done.add(r['task_name'])
rerun={'db-wal-recovery','path-tracing-reverse','torch-pipeline-parallelism'}
for t in open(f'{T}/tasks.txt').read().split():
    if t not in done or t in rerun: print(t)
EOF
log "maya-b tasks: $(tr '\n' ' ' < $T/tasks-maya-b.txt)"

llm-serve --stop 8000; sleep 8; OPENCODE_SYNC=0 llm-serve -d deepseek-maya
wait_up 8000 || { log "Maya did not come up"; llm-serve -d qwen; exit 1; }
hb tb20-maya-b 8000 $T/tasks-maya-b.txt
llm-serve --stop 8000; sleep 8

# Basalt: all 20, the llm-serve deepseek profile as adopted
OPENCODE_SYNC=0 llm-serve -d deepseek
if wait_up 8000; then hb tb20-basalt 8000 $T/tasks.txt; else log "Basalt did not come up"; fi
llm-serve --stop 8000; sleep 8

llm-serve -d qwen; wait_up 8000 && log "qwen back on :8000"
log "night run done"
