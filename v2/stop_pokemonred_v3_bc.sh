#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/PokemonRedExperiments-v2/v2"

BC_SOCKET="$HOME/PokemonRedExperiments-v2/v2/runs/bc_manager.sock"

echo "===================================================="
echo " Pokemon Red AI V3 BC - VEILIG STOPPEN"
echo "===================================================="

PID="$(
  ps -eo pid=,comm=,args= |
  awk '$2 ~ /^python/ && $0 ~ /baseline_fast_v2_bc\.py/ {print $1; exit}'
)"

SAVE_BEFORE=$(grep -c '\[V3-BC\] Ctrl-C checkpoint saved:' pokemon_training.log 2>/dev/null || true)

if [[ -n "$PID" ]]; then
    echo "PPO PID: $PID"
    kill -INT "$PID"
    SAVED=0
    for _ in $(seq 1 180); do
        SAVE_NOW=$(grep -c '\[V3-BC\] Ctrl-C checkpoint saved:' pokemon_training.log 2>/dev/null || true)
        if [[ "$SAVE_NOW" -gt "$SAVE_BEFORE" ]]; then
            echo "OK: actuele PPO-policy opgeslagen."
            SAVED=1
            break
        fi
        sleep 1
    done

    if [[ "$SAVED" != "1" ]]; then
        echo "FOUT: geen nieuwe Ctrl-C checkpoint-save gezien."
        echo "NIETS hard killen. Stuur deze output door."
        exit 1
    fi

    if kill -0 "$PID" 2>/dev/null; then
        PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
        echo "Cleanup procesgroep: $PGID"
        kill -TERM -- "-$PGID" 2>/dev/null || true
        sleep 5
        if kill -0 "$PID" 2>/dev/null; then
            kill -KILL -- "-$PGID" 2>/dev/null || true
        fi
    fi
else
    echo "PPO draaide al niet."
fi

tmux kill-session -t pokemonred 2>/dev/null || true
tmux kill-session -t pokemonlive 2>/dev/null || true
tmux kill-session -t tensorboard 2>/dev/null || true
tmux kill-session -t pokemonbctb 2>/dev/null || true
pkill -f '[b]c_tensorboard_history.py' 2>/dev/null || true
pkill -f '[l]ive_server.py' 2>/dev/null || true
pkill -f '[t]ensorboard' 2>/dev/null || true

# Ask the manager to flush/persist and exit gracefully.
python bc_manager.py --shutdown --socket "$BC_SOCKET" 2>/dev/null || true
for _ in $(seq 1 50); do
    [[ ! -S "$BC_SOCKET" ]] && break
    sleep 0.1
done
tmux kill-session -t pokemonbc 2>/dev/null || true
pkill -f '[b]c_manager.py --serve' 2>/dev/null || true

echo "OK: BC experiment gestopt."
echo "BC-memory blijft bewaard in runs/breadcrumb_routes_v3_ram.json"
