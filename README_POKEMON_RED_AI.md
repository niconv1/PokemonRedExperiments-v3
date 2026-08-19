# Pokémon Red AI V3 — lokale aangepaste versie

De publieke naam blijft **Pokémon Red AI V3**.

## Architectuur

```text
Agent 1
  = persistent
  = read-only voor PPO
  = eigen proces
  = automatische snelheidsbalans
  = GEEN data naar PPO

Agents 2..N
  = de enige PPO-trainingagents
```

Standaard:

```text
12 totaal = 1 explorer + 11 PPO trainers
```

Later bijvoorbeeld:

```text
20 totaal = 1 explorer + 19 PPO trainers
```

## Volledig clean geheugen

Voor een echte start vanaf nul worden verwijderd:

```text
PPO checkpoints
Agent 1 persistent state/meta
TensorBoard-runs
policy-sync snapshots
live frames
trainingslogs
```

`init.state` blijft behouden als schone beginstate.

## Start

```bash
cd ~/PokemonRedExperiments-v2/v2
./start_pokemonred_v3_2.sh
```

20 totaal:

```bash
./start_pokemonred_v3_2.sh 20
```

Automatisch op RAM:

```bash
./start_pokemonred_v3_2.sh auto
```

De helperbestandsnamen bevatten intern nog `v3_2`, maar de projectnaam blijft **V3**.

## Credits

Upstream:

```text
https://github.com/PWhiddy/PokemonRedExperiments
```

PWhiddy / Peter Whidden, MIT.
