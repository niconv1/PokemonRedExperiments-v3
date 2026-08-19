# Pokémon Red AI V3 — uitgebreide technische README

## 1. Naamgeving

De publieke projectnaam blijft:

```text
Pokémon Red AI V3
```

Interne helperbestandsnamen kunnen nog `v3_2` bevatten omdat de auto-balancer later binnen dezelfde V3-lijn is toegevoegd.

De upstream `v2/` directory en `_v2.py` bestandsnamen blijven behouden voor compatibiliteit.

## 2. Datastroom

```text
Agents 2..N -> PPO rollout buffer -> PPO update -> gedeelde policy
                                                     |
                                                     v
                                               Agent 1
                                               read-only
                                               eigen proces

Agent 1 transitions --------X--------> PPO
```

## 3. Agent 1

Agent 1:

- gebruikt `model.predict()`;
- heeft een eigen Game Boy environment;
- is persistent;
- schrijft niets naar PPO;
- heeft automatische pacing zodat hij ongeveer gelijkloopt met één PPO trainer.

## 4. Auto-balancer

De balancer vergelijkt Agent 1 met de voortgang van één PPO trainer:

```text
ppo_virtual_steps =
    (current_total_timesteps - start_total_timesteps)
    / num_train_envs
```

Doel:

```text
allowed_worker_steps =
    ppo_virtual_steps * target_ratio + lead_steps
```

Standaard:

```text
target_ratio = 1.0
```

Voorlopen -> throttle/sleep.
Achterlopen -> volle snelheid.

## 5. Dynamisch aantal agents

```text
num_train_envs = total_agents - 1
```

Voorbeelden:

```text
12 totaal -> 11 PPO trainers
20 totaal -> 19 PPO trainers
```

Agent 1 blijft altijd apart.

## 6. Volledig clean geheugen

Voor een echte reset naar nul worden alle runtime/trainingbestanden uit `v2/runs/` verwijderd behalve `.gitignore`.

Daarmee verdwijnen:

```text
PPO checkpoints
persistent Agent 1 state
persistent metadata
TensorBoard-runs
policy-sync snapshots
live JPEG/JSON
```

Ook trainingslogs worden gewist.

`init.state` blijft behouden en is de schone beginstate.

## 7. Start

```bash
./start_pokemonred_v3_2.sh
```

20 totaal:

```bash
./start_pokemonred_v3_2.sh 20
```

RAM-auto:

```bash
./start_pokemonred_v3_2.sh auto
```

## 8. Live viewer

Poort 8080.

Belangrijke velden:

```text
PPO FPS
Explorer FPS
Policy sync
Auto target
Ahead steps
Throttle
Flags
Last flag
Rewards
```

## 9. TensorBoard

Poort 6006.

Alle PPO-trainingdata komt alleen van Agents 2..N.

## 10. Git

Runtime/trained data hoort niet in Git:

```text
PokemonRed.gb
PPO checkpoints
persistent state
TensorBoard
policy snapshots
live frames
trainingslogs
lokale backups
Python caches
```

## 11. Credits

Upstream:

```text
https://github.com/PWhiddy/PokemonRedExperiments
```

PWhiddy / Peter Whidden, MIT.
