# Pokémon Red AI V3

**Pokémon Red AI V3** is a custom reinforcement-learning experiment based on
[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments).

The upstream source layout (`v2/`, `baseline_fast_v2.py`, `red_gym_env_v2.py`) is intentionally retained for compatibility.
The local project/release name is simply **V3**.

## Architecture

```text
                         Shared PPO policy
                               |
               +---------------+----------------+
               |                                |
               v                                v
       PPO TRAINING PATH                  READ-ONLY PATH
         Agents 2..N                        Agent 1
      N-1 PPO trainers              persistent explorer
               |                                |
  observations/actions/rewards                  | model.predict()
               |                                v
               v                         own Game Boy world
        PPO rollout buffer                own flags/rewards
               |                         persistent state
               v                                |
          PPO updates                             X
                                      NO DATA BACK INTO PPO
```

### Agent 1

Agent 1:

- is persistent;
- is read-only for PPO;
- runs in its own process;
- uses the current shared policy through `model.predict()`;
- never writes observations, actions, rewards, flags or transitions to PPO;
- is automatically paced to approximately the speed of one PPO trainer.

### Agents 2..N

All other agents are the only PPO training agents.

Default setup:

```text
12 total agents
1 read-only persistent explorer
11 PPO trainers
```

Example with more RAM:

```text
20 total agents
1 read-only persistent explorer
19 PPO trainers
```

## Automatic speed balancing

Agent 1 is automatically throttled when it runs ahead of one PPO trainer and allowed to run full speed when it falls behind.

This scales automatically with the number of PPO trainers.

## Start

Default:

```bash
cd v2
./start_pokemonred_v3_2.sh
```

20 total agents:

```bash
./start_pokemonred_v3_2.sh 20
```

RAM-based automatic choice:

```bash
./start_pokemonred_v3_2.sh auto
```

The helper filenames still contain `v3_2` internally because they were created during development, but the public project/release name remains **Pokémon Red AI V3**.

## Clean start

A completely clean run means:

```text
PPO model         = new
PPO steps         = 0
Agent 1 world     = init.state
Agents 2..N       = init.state
old checkpoints   = removed
old TensorBoard   = removed
old Agent 1 state = removed
old policy sync   = removed
old live metadata = removed
```

`init.state` remains because it is the intended clean Game Boy starting state.

## Live viewer

Default:

```text
http://<rig-ip>:8080
```

Shows PPO FPS, Explorer FPS, policy sync, auto target, ahead steps, throttle, flags and rewards.

## TensorBoard

Default:

```text
http://<rig-ip>:6006
```

Only PPO training agents contribute training data.

## Important source files

```text
v2/baseline_fast_v2.py
v2/red_gym_env_v2.py
v2/persistent_explorer_callback.py
v2/tensorboard_callback.py
v2/live_server.py
v2/install_pokemonred_v3_2_auto_speed.sh
v2/start_pokemonred_v3_2.sh
v2/stop_pokemonred_v3_2.sh
```

## Git hygiene

The repository excludes runtime/trained data:

```text
PokemonRed.gb
PPO checkpoints
persistent Agent 1 state
TensorBoard runs
policy-sync snapshots
live frames
training logs
local backups
Python caches
```

## Credits

Derived from:

**PWhiddy / Peter Whidden — PokemonRedExperiments**

https://github.com/PWhiddy/PokemonRedExperiments

Upstream license: MIT.

The original upstream README is preserved as `README_UPSTREAM.md`.
