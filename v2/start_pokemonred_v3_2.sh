#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/PokemonRedExperiments-v2/v2"
cd "$BASE"

# Usage:
#   ./start_pokemonred_v3_2.sh       -> 12 total agents
#   ./start_pokemonred_v3_2.sh 20    -> 20 total agents (1 explorer + 19 PPO)
#   ./start_pokemonred_v3_2.sh auto  -> 20 on >=28 GiB RAM, otherwise 12
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
    echo "Voor 20 agents vereist deze helper minimaal ~28 GiB gedetecteerd RAM."
    echo "Override alleen bewust met: POKEMON_FORCE_AGENTS=1"
    exit 1
fi

TARGET_RATIO="${AGENT1_SPEED_RATIO:-1.0}"
LEAD_STEPS="${AGENT1_LEAD_STEPS:-128}"

echo "Pokemon Red AI V3.2"
echo "RAM gedetecteerd : ~${MEM_GIB} GiB"
echo "Totaal agents    : ${TOTAL_AGENTS}"
echo "PPO trainers     : $((TOTAL_AGENTS - 1))"
echo "Agent 1 target   : ${TARGET_RATIO}x snelheid van 1 PPO trainer"
echo "Lead buffer      : ${LEAD_STEPS} steps"

tmux kill-session -t pokemonred 2>/dev/null || true
tmux kill-session -t pokemonlive 2>/dev/null || true
tmux kill-session -t tensorboard 2>/dev/null || true
pkill -f '[b]aseline_fast_v2.py' 2>/dev/null || true
pkill -f '[l]ive_server.py' 2>/dev/null || true
pkill -f '[t]ensorboard' 2>/dev/null || true

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
    cp -a pokemon_training.log "pokemon_training_before_v32_start_${STAMP}.log" || true
: > pokemon_training.log

if [[ -n "$LATEST" ]]; then
    CHECKPOINT="${LATEST%.zip}"
    echo "PPO hervatten vanaf: $LATEST"
    TRAIN_INNER="printf '%s\\n' '$CHECKPOINT' | python -u baseline_fast_v2.py"
else
    echo "Geen checkpoint: PPO start vanaf 0."
    TRAIN_INNER="python -u baseline_fast_v2.py"
fi

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; export POKEMON_TOTAL_AGENTS=$TOTAL_AGENTS; export AGENT1_SPEED_RATIO=$TARGET_RATIO; export AGENT1_LEAD_STEPS=$LEAD_STEPS; $TRAIN_INNER 2>&1 | tee pokemon_training.log; echo; echo TRAINING GESTOPT; exec bash'"

tmux new-session -d -s pokemonlive \
"cd /home/nico/PokemonRedExperiments-v2/v2 && /home/nico/miniconda3/envs/pokemonredv2/bin/python -u live_server.py"

tmux new-session -d -s tensorboard \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; tensorboard --logdir runs --host 0.0.0.0 --port 6006'"

echo
echo "Gestart."
echo "Live:        http://192.168.129.83:8080"
echo "TensorBoard: http://192.168.129.83:6006"
