#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/PokemonRedExperiments-v2/v2"

echo "V3.2 netjes stoppen..."
tmux send-keys -t pokemonred C-c 2>/dev/null || true

for _ in $(seq 1 45); do
    if ! pgrep -f '[b]aseline_fast_v2.py' >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if pgrep -f '[b]aseline_fast_v2.py' >/dev/null 2>&1; then
    echo "WAARSCHUWING: training reageerde niet; beëindigen."
    pkill -f '[b]aseline_fast_v2.py' 2>/dev/null || true
    sleep 2
fi

tmux kill-session -t pokemonred 2>/dev/null || true
tmux kill-session -t pokemonlive 2>/dev/null || true
tmux kill-session -t tensorboard 2>/dev/null || true
pkill -f '[l]ive_server.py' 2>/dev/null || true
pkill -f '[t]ensorboard' 2>/dev/null || true

echo "V3.2 gestopt."
