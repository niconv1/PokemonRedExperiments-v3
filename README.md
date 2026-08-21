# Pokémon Red AI V3

**Pokémon Red AI V3** is a custom reinforcement-learning experiment based on
[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments).

The upstream source layout (`v2/`, `baseline_fast_v2.py`, `red_gym_env_v2.py`) is intentionally retained for compatibility.
The public project/release name remains simply **V3**.

## What V3 adds

V3 separates exploration from PPO training and adds a lightweight shared breadcrumb-learning system.

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
                  |                                | model.predict()
                  |                                v
                  |                         own Game Boy world
                  |                         persistent state
                  |                                |
                  |                                X
                  |                     NO DATA BACK INTO PPO
                  |
                  +-------------------+
                                      |
                                      v
                         lightweight BC clients
                                      |
                         rare discovery datagrams
                                      |
                                      v
                         CENTRAL RAM BC MANAGER
                         ----------------------
                         shortest observed route
                         target mastery / decay
                         atomic persistence
                         RAM snapshot in /dev/shm
                                      |
                                      v
                          read-only route snapshot
                                      |
                                      +----> PPO agents
                                      |
                                      +----> TensorBoard BC tab
```

## Agent 1: persistent read-only explorer

Agent 1:

- is persistent;
- runs in its own process;
- uses the current shared policy through `model.predict()`;
- keeps its own Game Boy world and persistent state;
- never writes observations, actions, rewards, flags or transitions into the PPO rollout buffer;
- does not write breadcrumb discoveries;
- can continue exploring while Agents 2..N perform PPO training.

## Agents 2..N: PPO trainers

All other agents are the PPO training agents.

Default setup:

```text
12 total agents
1 persistent read-only explorer
11 PPO trainers
```

The number of total agents can be changed depending on available CPU and RAM.

## Central RAM Breadcrumb Learning

The BC system learns routes automatically from successful PPO agents.

A breadcrumb target is a real named event flag, for example:

```text
Entered Blues House
Got Potion Sample
Got Oaks Parcel
Got Pokedex
```

The manager stores the map-transition route that preceded a successful discovery.

Example:

```text
Flag: Entered Blues House
Route: 38 -> 37 -> 0 -> 37 -> 0 -> 39
```

### Single-writer design

All mutable breadcrumb knowledge lives in one central manager process.

PPO workers do **not** rewrite a shared JSON database and do not take a shared file lock on every step.

Instead:

1. A PPO worker sends a tiny non-blocking Unix datagram only when it discovers a real named target.
2. The central BC manager updates the route database.
3. A compact read-only snapshot is published in `/dev/shm`.
4. PPO workers only check BC route progress when the Game Boy map actually changes.
5. Persistent JSON is written atomically by the manager.

This keeps BC out of the normal PPO hot path as much as possible.

## Shortest observed route

For each target, V3 keeps the shortest route that has actually been observed successfully.

```text
new route shorter than current route  -> replace
same length                            -> keep existing route
longer route                           -> ignore
```

A target being mastered does **not** freeze its route.

If a shorter successful route is discovered later, the mastered target is still updated to the shorter route.

## Mastery

Breadcrumb reward decays as more distinct PPO agents independently reach the target:

```text
1 successful agent  -> mastery 1.00
2 successful agents -> mastery 0.60
3 successful agents -> mastery 0.30
4+ agents            -> mastery 0.00 / MASTERED
```

`MASTERED` means the target no longer needs extra breadcrumb reward.

It does **not** mean route learning stops.

A mastered route can still be improved when a shorter successful path is discovered.

## Breadcrumb reward

The BC client uses a bounded route reward budget.

By default:

```text
full BC route budget <= 0.25
```

Multiple matching routes do not stack unbounded reward on a single map transition.

Mastered targets have a mastery factor of `0.00` and therefore provide no further BC reward.

## TensorBoard BC tab

V3 adds a dedicated **BC** tab to TensorBoard.

Default TensorBoard URL:

```text
http://<rig-ip>:6006
```

The top navigation contains:

```text
TIME SERIES | SCALARS | IMAGES | DISTRIBUTIONS | HISTOGRAMS | TEXT | BC
```

The BC tab shows:

- live BC version;
- total targets;
- active targets;
- mastered targets;
- discovery count;
- target/flag names;
- shortest known route;
- number of successful agents;
- mastery state.

Example:

```text
step 3,604,480

Flag 1  Entered Blues House
Route: 38 -> 37 -> 0 -> 37 -> 0 -> 39
Agents: 3
Mastery: 30%
```

## BC history aligned with TensorBoard TEXT

A small watcher process observes:

```text
trajectory/all_flags/text_summary
```

Whenever TensorBoard receives a new TEXT summary, the current BC snapshot is stored with the **same TensorBoard step number**.

This makes it possible to compare normal trajectory text and breadcrumb knowledge at matching training steps.

Runtime history is stored locally and excluded from Git.

## Performance

The central-RAM implementation was built specifically to avoid the heavy slowdown of earlier file-based breadcrumb experiments.

Reference test:

```text
CPU              Intel Core i7-7700
GPU              NVIDIA GeForce RTX 3060 Ti
Total agents     12
PPO trainers     11
Agent 1          persistent read-only
BC manager       central RAM single-writer
Observed PPO FPS ~552
```

This is a reference measurement, not a guaranteed benchmark for other hardware.

## Start V3 with Central RAM BC

```bash
cd v2
./start_pokemonred_v3_bc.sh
```

The starter launches:

```text
pokemonred   -> PPO training
pokemonbc    -> central BC manager
pokemonbctb  -> TensorBoard BC history watcher
pokemonlive  -> live viewer
tensorboard  -> TensorBoard
```

## Check BC status

```bash
cd v2
./check_pokemonred_v3_bc.sh
```

This reports:

```text
PPO FPS
timesteps
iterations
BC version
targets
active/mastered targets
discoveries
routes
processes
tmux sessions
```

## Stop V3 with BC

```bash
cd v2
./stop_pokemonred_v3_bc.sh
```

## Stable non-BC reference

The earlier V3 reference path is still retained:

```bash
cd v2
./start_pokemonred_v3_2.sh
```

The helper filename contains `v3_2` for historical development reasons; the public release name is still **Pokémon Red AI V3**.

## Live viewer

Default:

```text
http://<rig-ip>:8080
```

The live viewer displays the selected agent, PPO/explorer statistics, flags and rewards.

## Important source files

Core V3:

```text
v2/baseline_fast_v2.py
v2/red_gym_env_v2.py
v2/persistent_explorer_callback.py
v2/tensorboard_callback.py
v2/live_server.py
```

Central RAM BC:

```text
v2/baseline_fast_v2_bc.py
v2/red_gym_env_v2_bc.py
v2/bc_manager.py
v2/test_bc_manager.py
v2/start_pokemonred_v3_bc.sh
v2/stop_pokemonred_v3_bc.sh
v2/check_pokemonred_v3_bc.sh
```

TensorBoard BC integration:

```text
v2/bc_tensorboard_history.py
v2/tensorboard_bc_plugin/setup.py
v2/tensorboard_bc_plugin/pokemonred_tensorboard_bc/plugin.py
```

## BC self-test

```bash
cd v2
python test_bc_manager.py
```

The test covers:

```text
central single-writer manager
Unix datagram discovery events
RAM snapshot
shortest observed route
mastery 1.00 -> 0.60 -> 0.30 -> 0.00
bounded total route reward
mastered target reward = 0
Agent 1 read-only behavior
atomic persistence
```

## Runtime data and Git hygiene

Runtime/trained data is intentionally excluded from Git, including:

```text
Pokemon ROMs
PPO checkpoints
persistent Agent 1 state
TensorBoard event data
policy-sync snapshots
live frames
training logs
BC runtime JSON
BC TensorBoard history JSONL
Python caches / egg-info
local experiment backups
```

Do not commit a Pokémon ROM or trained runtime state to this repository.

## Credits

Derived from:

**PWhiddy / Peter Whidden — PokemonRedExperiments**

Upstream project:

https://github.com/PWhiddy/PokemonRedExperiments

Upstream license: MIT.

The original upstream README is preserved as `README_UPSTREAM.md`.
