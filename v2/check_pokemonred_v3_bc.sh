#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/PokemonRedExperiments-v2/v2"

SNAP="/dev/shm/pokemonred_bc_${UID}.json"

echo "===================================================="
echo " Pokemon Red AI V3 - BC STATUS"
echo "===================================================="

echo
echo "===== PPO LAATSTE METRICS ====="
python - <<'PY'
from pathlib import Path
import re
p = Path('pokemon_training.log')
text = p.read_text(errors='replace') if p.exists() else ''
fps = re.findall(r'\|\s+fps\s+\|\s+([0-9]+)', text)
steps = re.findall(r'\|\s+total_timesteps\s+\|\s+([0-9]+)', text)
iters = re.findall(r'\|\s+iterations\s+\|\s+([0-9]+)', text)
print('PPO FPS       :', fps[-1] if fps else '-')
print('Timesteps     :', steps[-1] if steps else '-')
print('Iterations    :', iters[-1] if iters else '-')
PY

echo
echo "===== BC MANAGER ====="
python - "$SNAP" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print('BC snapshot ontbreekt:', p)
    raise SystemExit(0)
d = json.loads(p.read_text())
print('Version       :', d.get('version', '-'))
print('Targets       :', d.get('total_targets', 0))
print('Active        :', d.get('active_targets', 0))
print('Mastered      :', d.get('mastered_targets', 0))
print('Discoveries   :', d.get('total_discoveries', 0))
print()
items = list(d.get('targets', {}).values())
items.sort(key=lambda x: (-int(x.get('successful_agent_count', 0)), len(x.get('route', [])), x.get('target','')))
for r in items[:12]:
    print(
        f"- {r.get('target','?')} | route={r.get('route', [])} | "
        f"agents={r.get('successful_agent_count',0)} | "
        f"mastery={float(r.get('mastery_factor',0)):.2f}"
    )
PY

echo
echo "===== PROCESSEN ====="
ps -eo pid,ppid,stat,%cpu,%mem,args --sort=-%cpu | \
grep -E 'baseline_fast_v2_bc.py|bc_manager.py --serve|pokemon-agent1|multiprocessing.forkserver|multiprocessing.spawn' | \
grep -v grep | head -30 || true

echo
echo "===== TMUX ====="
tmux ls 2>/dev/null || true
