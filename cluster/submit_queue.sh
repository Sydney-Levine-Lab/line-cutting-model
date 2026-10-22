#!/bin/bash
# Submit sbatch arrays one at a time, waiting for room under the per-user submit cap.
# Run on the login node, from the repo root:
#   nohup bash cluster/submit_queue.sh cluster/queue.txt > logs/submit_queue.log 2>&1 &
# queue.txt: one submission per line, exactly as typed at the prompt; blank lines and '#' comments ignored.
# Counts queued TASKS (squeue -r), since the cap counts array tasks, not arrays. Already-complete cells skip anyway.
CAP=${CAP:-440}; NEED=${NEED:-140}
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  while (( $(squeue -u "$USER" -h -r | wc -l) + NEED > CAP )); do sleep 300; done
  echo "$(date '+%F %T') submitting: $line"
  eval "$line" || echo "FAILED: $line"
  sleep 60
done < "$1"
echo "$(date '+%F %T') queue done"
