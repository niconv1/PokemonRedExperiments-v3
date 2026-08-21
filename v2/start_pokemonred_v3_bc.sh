#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/PokemonRedExperiments-v2/v2"
cd "$BASE"

# Usage:
#   ./start_pokemonred_v3_bc.sh       -> 12 total agents
#   ./start_pokemonred_v3_bc.sh 20    -> 20 total agents (with sufficient RAM)
#
# BC tuning (defaults preserve the Phase-1 design):
#   BC_REWARD_BUDGET=0.25
#   BC_TRACE_LEN=32

REQUESTED="${1:-${POKEMON_TOTAL_AGENTS:-12}}"
MEM_KIB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
MEM_GIB=$(( MEM_KIB / 1024 / 1024 ))

if [[ "$REQUESTED" == "auto" ]]; then
    if (( MEM_GIB >= 28 )); then
        TOTAL_AGENTS=20
    else
        TOTAL_AGENTS=12
    fi
else
    TOTAL_AGENTS="$REQUESTED"
fi

if ! [[ "$TOTAL_AGENTS" =~ ^[0-9]+$ ]] || (( TOTAL_AGENTS < 2 )); then
    echo "FOUT: totaal aantal agents moet een getal >= 2 zijn."
    exit 1
fi

if (( TOTAL_AGENTS >= 20 && MEM_GIB < 28 )) && [[ "${POKEMON_FORCE_AGENTS:-0}" != "1" ]]; then
    echo "FOUT: $TOTAL_AGENTS agents gevraagd met slechts ~${MEM_GIB} GiB RAM."
    echo "Override alleen bewust met: POKEMON_FORCE_AGENTS=1"
    exit 1
fi

TARGET_RATIO="${AGENT1_SPEED_RATIO:-1.0}"
LEAD_STEPS="${AGENT1_LEAD_STEPS:-128}"
BC_REWARD_BUDGET="${BC_REWARD_BUDGET:-0.25}"
BC_TRACE_LEN="${BC_TRACE_LEN:-32}"
BC_SOCKET="$BASE/runs/bc_manager.sock"
BC_SNAPSHOT="/dev/shm/pokemonred_bc_${UID}.json"
BC_PERSIST="$BASE/runs/breadcrumb_routes_v3_ram.json"

echo "===================================================="
echo " Pokemon Red AI V3 - CENTRAL RAM BC EXPERIMENT"
echo "===================================================="
echo "RAM gedetecteerd : ~${MEM_GIB} GiB"
echo "Totaal agents    : ${TOTAL_AGENTS}"
echo "PPO trainers     : $((TOTAL_AGENTS - 1))"
echo "Agent 1 target   : ${TARGET_RATIO}x per PPO trainer"
echo "BC reward budget : ${BC_REWARD_BUDGET} vóór reward_scale"
echo "BC trace length  : ${BC_TRACE_LEN} map transitions"
echo "BC persistence   : ${BC_PERSIST}"
echo

# Clean old runtime processes/sessions, but NEVER delete BC persistence,
# checkpoints, Agent-1 world state, or TensorBoard history.
tmux kill-session -t pokemonred 2>/dev/null || true
tmux kill-session -t pokemonlive 2>/dev/null || true
tmux kill-session -t tensorboard 2>/dev/null || true
tmux kill-session -t pokemonbc 2>/dev/null || true
tmux kill-session -t pokemonbctb 2>/dev/null || true
pkill -f '[b]c_tensorboard_history.py' 2>/dev/null || true
pkill -f '[b]aseline_fast_v2_bc.py' 2>/dev/null || true
pkill -f '[b]aseline_fast_v2.py' 2>/dev/null || true
pkill -f '[l]ive_server.py' 2>/dev/null || true
pkill -f '[t]ensorboard' 2>/dev/null || true
pkill -f '[b]c_manager.py --serve' 2>/dev/null || true

rm -f "$BC_SOCKET" "$BC_SNAPSHOT"
rm -f runs/curframe_*.jpeg runs/curframe_*.json
rm -f runs/persistent_agent_1_worker_status.json
rm -f runs/persistent_agent_1_policy.pt
rm -f runs/persistent_agent_1_policy.pt.tmp
rm -f runs/persistent_agent_1_policy_bootstrap.zip
rm -f runs/persistent_agent_1_policy_bootstrap.tmp.zip

LATEST="$(
python - <<'PY'
from pathlib import Path
import re
best = None
for p in Path("runs").glob("poke_*_steps.zip"):
    m = re.fullmatch(r"poke_(\d+)_steps\.zip", p.name)
    if m:
        item = (int(m.group(1)), p)
        if best is None or item[0] > best[0]:
            best = item
if best:
    print(best[1])
PY
)"

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ -f pokemon_training.log ]] && \
    cp -a pokemon_training.log "pokemon_training_before_bc_${STAMP}.log" || true
[[ -f bc_manager.log ]] && \
    cp -a bc_manager.log "bc_manager_before_${STAMP}.log" || true
: > pokemon_training.log
: > bc_manager.log

# Start central single-writer BC manager first.
tmux new-session -d -s pokemonbc \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; python -u bc_manager.py --serve --socket "$BC_SOCKET" --snapshot "$BC_SNAPSHOT" --persist "$BC_PERSIST" 2>&1 | tee bc_manager.log; echo; echo BC MANAGER GESTOPT; exec bash'"

READY=0
for _ in $(seq 1 100); do
    if [[ -S "$BC_SOCKET" && -f "$BC_SNAPSHOT" ]]; then
        READY=1
        break
    fi
    sleep 0.05
done

if [[ "$READY" != "1" ]]; then
    echo "FOUT: BC manager werd niet klaar."
    tmux capture-pane -pt pokemonbc 2>/dev/null | tail -40 || true
    exit 1
fi

echo "BC manager: READY"

if [[ -n "$LATEST" ]]; then
    CHECKPOINT="${LATEST%.zip}"
    echo "PPO hervatten vanaf: $LATEST"
    TRAIN_INNER="printf '%s\\n' '$CHECKPOINT' | python -u baseline_fast_v2_bc.py"
else
    echo "Geen checkpoint: PPO start vanaf 0."
    TRAIN_INNER="python -u baseline_fast_v2_bc.py"
fi

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; export POKEMON_TOTAL_AGENTS=$TOTAL_AGENTS; export AGENT1_SPEED_RATIO=$TARGET_RATIO; export AGENT1_LEAD_STEPS=$LEAD_STEPS; export POKEMON_BC_ENABLED=1; export POKEMON_BC_SOCKET_PATH="$BC_SOCKET"; export POKEMON_BC_SNAPSHOT_PATH="$BC_SNAPSHOT"; export POKEMON_BC_REWARD_BUDGET=$BC_REWARD_BUDGET; export POKEMON_BC_TRACE_LEN=$BC_TRACE_LEN; $TRAIN_INNER 2>&1 | tee pokemon_training.log; echo; echo TRAINING GESTOPT; exec bash'"

tmux new-session -d -s pokemonlive \
"cd /home/nico/PokemonRedExperiments-v2/v2 && /home/nico/miniconda3/envs/pokemonredv2/bin/python -u live_server.py"

tmux new-session -d -s pokemonbctb \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; python -u bc_tensorboard_history.py --logdir runs --snapshot /dev/shm/pokemonred_bc_${UID}.json --history runs/bc_tensorboard_history.jsonl --poll 3 2>&1 | tee bc_tensorboard_history.log; echo; echo BC TENSORBOARD HISTORY GESTOPT; exec bash'"

tmux new-session -d -s tensorboard \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; tensorboard --logdir runs --host 0.0.0.0 --port 6006'"

echo
echo "Gestart."
echo "Live:        http://192.168.129.83:8080"
echo "TensorBoard: http://192.168.129.83:6006  (extra BC tab)"
echo "BC snapshot: $BC_SNAPSHOT"
echo
echo "Na 3-5 minuten: ./check_pokemonred_v3_bc.sh"
