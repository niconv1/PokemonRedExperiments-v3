# Pokémon Red AI V3

**Pokémon Red AI V3** is a custom reinforcement-learning experiment based on
[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments).

V3 keeps the original PyBoy + Stable-Baselines3 PPO foundation, but changes the training architecture so one agent can act as a long-running explorer **without contaminating PPO training data**.

> This repository is an independent experimental fork / derivative project.
> The original project remains credited to PWhiddy / Peter Whidden and is available at the link above.

---

## What is new in V3?

The biggest change is the separation between the **PPO training agents** and a **persistent read-only explorer**.

```text
                         Shared PPO policy
                               |
               +---------------+---------------+
               |                               |
               v                               v
        Agents 2–12                         Agent 1
      11 PPO trainers              Persistent read-only explorer
               |                               |
    observations/actions/rewards               | model.predict()
               |                               v
               v                         own Game Boy world
        PPO rollout buffer                own flags/rewards
               |                         persistent state
               v                               |
          PPO updates                            X
                                     NO DATA BACK INTO PPO
```

### Agent 1 — persistent read-only explorer

Agent 1:

- reads the latest shared PPO policy with `model.predict()`;
- uses its own independent Pokémon Red environment;
- keeps its own persistent Game Boy world state;
- continues exploring instead of restarting at the normal `max_steps` boundary;
- can keep its step, heal, exploration and reward counters running during the process;
- periodically autosaves its world state;
- **does not add observations, actions, rewards or transitions to the PPO rollout buffer**;
- therefore **cannot directly train or distort PPO**.

The knowledge flow is intentionally one-way:

```text
Agents 2–12 -> PPO learning -> newer shared policy -> Agent 1
```

not:

```text
Agent 1 -> PPO training
```

### Agents 2–12 — the only PPO trainers

Agents 2 through 12 are the **11 training environments**.

They:

- collect PPO rollouts;
- generate the training rewards;
- update the policy/value network;
- use ordinary episode/reset behavior.

Agent 1 is not part of this vectorized PPO environment.

---

## Why this architecture?

A persistent explorer can reach locations and game states far beyond a normal training episode.

If that persistent environment also feeds its transitions into PPO, it can create a very different data distribution from the ordinary reset agents.

V3 separates these responsibilities:

- **Agents 2–12:** learn the policy.
- **Agent 1:** uses that policy to explore a continuing world.

This keeps the persistent experiment useful while keeping PPO training data controlled.

---

## Core V3 architecture

```text
baseline_fast_v2.py
        |
        +---- SubprocVecEnv
        |       |
        |       +-- Agent 2
        |       +-- Agent 3
        |       +-- ...
        |       +-- Agent 12
        |               |
        |               v
        |          PPO rollout buffer
        |               |
        |               v
        |          PPO policy update
        |
        +---- PersistentExplorerCallback
                |
                +-- Agent 1
                +-- model.predict()
                +-- persistent world state
                +-- NO rollout-buffer writes
```

Important V3 files:

```text
v2/baseline_fast_v2.py
v2/red_gym_env_v2.py
v2/persistent_explorer_callback.py
v2/tensorboard_callback.py
v2/live_server.py
v2/install_pokemonred_v3_fresh_zero.sh
```

The `v2/` directory and some `_v2.py` filenames are retained for compatibility with the original source layout.
**The architecture/release in this repository is V3.**

---

## Current training configuration

Default V3 configuration:

```text
PPO training agents : 11  (Agents 2–12)
Persistent explorer : 1   (Agent 1)
ep_length           : 163840
n_steps             : 2560
action_freq         : 24
headless            : True
print_rewards       : False
reward_scale        : 0.5
explore_weight      : 0.25
```

One PPO rollout therefore contains:

```text
11 × 2560 = 28160 PPO timesteps
```

Agent 1 is deliberately excluded from that count.

---

## Fresh-zero training

V3 supports a true clean start:

```text
PPO model        = new
PPO steps        = 0
Agents 2–12      = from init.state
Agent 1 world    = from init.state
old checkpoints  = not loaded
old TensorBoard  = not loaded
old Agent 1 state = not loaded
```

The installer included in this repository is:

```text
v2/install_pokemonred_v3_fresh_zero.sh
```

It performs architecture checks and validates the V3 source before installation.

---

## Checkpoint resume behavior

When training resumes from a checkpoint, V3 uses:

```python
reset_num_timesteps=False
```

This applies to the **PPO model / Agents 2–12**, not to Agent 1.

As a result, the PPO `total_timesteps` counter can continue from the checkpoint instead of restarting at zero.

Agent 1 maintains its own separate environment step counter.

---

## Persistent Agent 1 state

Agent 1 stores its Game Boy world state in:

```text
v2/runs/persistent_agent_1.state
```

Runtime state, checkpoints and training output are intentionally excluded from Git.

The repository keeps `init.state`, because it is the clean starting Game Boy state required by the environment.

---

## Live viewer

V3 includes a local live viewer:

```text
v2/live_server.py
```

Default port:

```text
8080
```

It can display:

- selected agent;
- Agent 1 persistent/read-only status;
- FPS;
- agent steps;
- PPO steps;
- flags;
- last named flag;
- badges;
- heal reward;
- exploration reward;
- reward sum;
- done state.

Runtime JPEG/JSON frames are ignored by Git.

---

## TensorBoard

TensorBoard is normally served on:

```text
6006
```

V3 TensorBoard training metrics represent the PPO training path — **Agents 2–12**.

Agent 1 is not a PPO rollout source.

### Flag logging

The local flag-logging changes distinguish between:

- episode-local flags;
- current episode flags;
- optional cumulative discovered flags.

`Last flag` is viewer/debug metadata and is not PPO input.

---

## Event flag fixes

V3 includes local event-flag work such as LSB-first event-bit interpretation:

```python
for idx in range(8):
    if val & (1 << idx):
        ...
```

This maps event names to the intended Game Boy event-bit positions.

---

## Checkpoints and runtime files are not stored in Git

The V3 repository intentionally excludes:

```text
PokemonRed.gb
persistent Agent 1 runtime state
PPO checkpoint ZIP files
TensorBoard event/runs
live JPEG/JSON frames
training logs
local backups
Python caches
```

This keeps the Git repository focused on source code and reproducible configuration rather than trained model history.

---

## ROM

A Pokémon Red ROM is **not included**.

The environment expects a legally obtained ROM named:

```text
PokemonRed.gb
```

in the project root.

The ROM used during local development matched SHA-1:

```text
ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

---

## Running V3 locally

The source currently retains the upstream directory layout.

Example:

```bash
cd v2
python baseline_fast_v2.py
```

For local development, dependencies are listed in:

```text
v2/requirements.txt
```

The exact CUDA/PyTorch installation can depend on the host GPU and driver setup.

---

## Additional documentation

More detailed local documentation is available in:

- [`README_POKEMON_RED_AI.md`](README_POKEMON_RED_AI.md)
- [`README_POKEMON_RED_AI_EXTENDED.md`](README_POKEMON_RED_AI_EXTENDED.md)
- [`README_UPSTREAM.md`](README_UPSTREAM.md) — preserved original upstream README after the V3 README migration

---

## V3 release

Initial clean V3 release:

```text
v3.0.0
```

Key V3 properties:

```text
Agent 1      = persistent read-only explorer
Agents 2–12  = 11 PPO training agents
PPO learning = Agents 2–12 only
Policy use   = shared
Agent 1 data = never written to PPO rollout buffer
```

---

## Credits

Pokémon Red AI V3 is derived from:

**PWhiddy / Peter Whidden — PokemonRedExperiments**

Original repository:

https://github.com/PWhiddy/PokemonRedExperiments

Original project license:

```text
MIT
```

This V3 repository keeps the upstream license and attribution while documenting the additional local experimental architecture separately.

---

## Project status

V3 is an experimental research / hobby project.

The current focus is:

1. stable PPO training with 11 reset-based trainers;
2. a completely separate persistent explorer;
3. clean checkpoint handling;
4. reliable live monitoring;
5. accurate event/flag progress logging;
6. long-running local training without mixing persistent explorer transitions into PPO.
