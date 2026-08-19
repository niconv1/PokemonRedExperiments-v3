# Pokémon Red AI V3 — lokale aangepaste versie

Deze repository/map bevat **Pokémon Red AI V3**, een lokale aangepaste voortzetting van het open-sourceproject:

- **GitHub:** https://github.com/PWhiddy/PokemonRedExperiments
- **Upstream:** PWhiddy / Peter Whidden
- **Upstream project:** *Train RL agents to play Pokemon Red*
- **Upstream licentie:** MIT
- **Belangrijkste upstream-technieken:** PyBoy, Stable-Baselines3 PPO, parallelle Gym environments en TensorBoard.

> Deze lokale versie is geen officiële upstream-release. Ze bevat eigen wijzigingen voor live monitoring, flags, checkpoints en een aparte persistente read-only explorer.

> **Naamgeving:** de lokale projectmap en enkele bronbestanden behouden voorlopig `v2` in hun naam voor compatibiliteit met de bestaande code. De **architectuur/release zelf heet V3**.

## Belangrijkste architectuur

Na succesvolle installatie van `install_pokemonred_v3_fresh_zero.sh` zijn er **12 zichtbare agents**, maar slechts **11 PPO-trainingagents**:

```text
                    PPO MODEL / POLICY
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Agents 2–12                 Agent 1
   11 PPO training envs     [PERSISTENT READ-ONLY]
             |                           |
             | observations             | model.predict()
             | actions                  |
             | rewards                  v
             v                    eigen Game Boy-wereld
       PPO rollout buffer         eigen flags/rewards
             |                    persistent autosave
             v                           |
         PPO update                       X
                                  GEEN DATA NAAR PPO
```

### Agent 1

Agent 1 is een **persistent read-only explorer**:

- gebruikt de actuele PPO-policy via `model.predict()`;
- profiteert dus van wat Agents 2–12 leren;
- zit **niet** in de PPO `SubprocVecEnv`;
- schrijft **geen observations, actions, rewards of flags** naar de PPO rollout-buffer;
- kan het PPO-model dus niet trainen of scheeftrekken;
- houdt zijn eigen Pokémon-wereld bij;
- gebruikt `runs/persistent_agent_1.state`;
- autosave blijft actief;
- heeft geen normale `max_steps` reset;
- `steps`, `heal`, `explore` en `reward sum` mogen tijdens het draaiende explorerproces blijven oplopen.

**Belangrijke beperking:** de Game Boy-worldstate blijft over een procesherstart behouden. Python-only tellers zoals cumulatieve heal/reward/steps worden momenteel niet als volledige countersnapshot over een volledige procesherstart bewaard.

### Agents 2–12

Agents 2–12 zijn de enige PPO-trainingagents:

- 11 parallelle training environments;
- normale V2 episode/resetlogica;
- hun ervaringen vullen de PPO rollout-buffer;
- alleen hun data beïnvloedt de PPO-policy.

## Installatiepaden

Project:

```text
/home/nico/PokemonRedExperiments-v2
```

V2:

```text
/home/nico/PokemonRedExperiments-v2/v2
```

Conda environment:

```text
pokemonredv2
```

ROM:

```text
/home/nico/PokemonRedExperiments-v2/PokemonRed.gb
```

Verwachte SHA-1:

```text
ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

De ROM wordt niet door dit project geleverd.

## Huidige hoofdconfiguratie

```text
PPO trainingagents : 11  (Agents 2–12)
Read-only explorer : 1   (Agent 1)
ep_length          : 163840
n_steps            : 2560
headless           : True
action_freq        : 24
print_rewards      : False
reward_scale       : 0.5
explore_weight     : 0.25
```

Eén PPO rollout bevat nu:

```text
11 × 2560 = 28160 PPO timesteps
```

Agent 1 telt hier bewust **niet** in mee.

## PPO checkpoints

Checkpoints staan in:

```text
/home/nico/PokemonRedExperiments-v2/v2/runs
```

Laatste checkpoint zoeken:

```bash
cd ~/PokemonRedExperiments-v2/v2
LATEST=$(find runs -maxdepth 1 -type f -name 'poke_*_steps.zip' | sort -V | tail -n 1)
echo "$LATEST"
```

Bij hervatten wordt `reset_num_timesteps=False` gebruikt.

Dat betekent:

- de PPO-modelgewichten worden geladen;
- de PPO `total_timesteps` hoort verder te tellen vanaf het checkpoint;
- deze PPO teller hoort bij **Agents 2–12 / het PPO-model**;
- hij is **niet** de step-teller van Agent 1.

Controle:

```bash
grep -i "loading checkpoint" pokemon_training.log
```

## Bot starten

Nieuwe training:

```bash
cd ~/PokemonRedExperiments-v2/v2

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; python -u baseline_fast_v2.py 2>&1 | tee pokemon_training.log; exec bash'"
```

Hervatten vanaf laatste checkpoint:

```bash
cd ~/PokemonRedExperiments-v2/v2

LATEST=$(find runs -maxdepth 1 -type f -name 'poke_*_steps.zip' | sort -V | tail -n 1)
CHECKPOINT="${LATEST%.zip}"

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; echo \"$CHECKPOINT\" | python -u baseline_fast_v2.py 2>&1 | tee pokemon_training.log; exec bash'"
```

## Bot stoppen

```bash
tmux send-keys -t pokemonred C-c 2>/dev/null || true
sleep 6

pkill -f '[b]aseline_fast_v2.py' 2>/dev/null || true
sleep 2

tmux kill-session -t pokemonred 2>/dev/null || true
```

Controle:

```bash
pgrep -af baseline_fast_v2.py || echo "Bot gestopt"
```

## Live viewer — poort 8080

Start:

```bash
cd ~/PokemonRedExperiments-v2/v2

tmux kill-session -t pokemonlive 2>/dev/null || true

tmux new-session -d -s pokemonlive \
"cd /home/nico/PokemonRedExperiments-v2/v2 && /home/nico/miniconda3/envs/pokemonredv2/bin/python -u live_server.py"
```

Open:

```text
http://192.168.129.83:8080
```

De viewer toont onder andere:

- Agent 1 als `[PERSISTENT READ-ONLY]`;
- agent steps;
- globale PPO steps;
- globale FPS;
- flags;
- Last flag;
- badges;
- heal;
- explore;
- rewards.

## TensorBoard — poort 6006

Start:

```bash
tmux kill-session -t tensorboard 2>/dev/null || true

tmux new-session -d -s tensorboard \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; tensorboard --logdir runs --host 0.0.0.0 --port 6006'"
```

Open:

```text
http://192.168.129.83:6006
```

De PPO/TensorBoard-trainingdata hoort nu alleen bij **Agents 2–12**.

Agent 1 is een read-only explorer en hoort niet in PPO rollout-, loss- of reward-trainingstatistieken terecht te komen.

## Flags

Lokale fixes:

- eventbitnamen worden LSB-first gedecodeerd;
- `Last flag` wordt chronologisch bijgehouden;
- persistent flagmetadata voor Agent 1 is alleen viewer/debuginformatie;
- `Last flag` beïnvloedt PPO niet.

TensorBoard:

- `trajectory/all_flags` = flags uit de betreffende voltooide PPO-episode;
- `trajectory/current_flags` = dezelfde episodecontext;
- `trajectory/ever_discovered_flags` = apart cumulatief logboek.

## Belangrijke lokale wijzigingen

Ten opzichte van upstream V2:

1. CUDA/PyTorch setup voor RTX 3060 Ti.
2. Worker count aangepast.
3. Live viewer op 8080.
4. TensorBoard op 6006.
5. Live JSON/JPEG metadata.
6. Eén agent tegelijk in viewer om I/O te beperken.
7. `print_rewards=False`.
8. Event-bitvolgorde gecorrigeerd naar LSB-first.
9. Chronologische Last flag tracking.
10. TensorBoard `all_flags` hersteld naar episode-lokale betekenis.
11. Persistent Agent 1 worldstate/autosave.
12. Agent 1 volledig uit PPO-training gehaald.
13. Agents 2–12 zijn nu de 11 PPO-trainingagents.
14. Agent 1 leest de actuele policy uitsluitend via `model.predict()`.
15. Agent 1 heeft geen normale `max_steps` reset.
16. Agent 1 zijn heal/explore/reward/steps lopen tijdens zijn proces door.
17. `reset_num_timesteps=False` bij checkpoint-resume.
18. Backup- en rollbacklogica toegevoegd aan de migratie.

## Migratie / backup

De nieuwe read-only architectuur wordt geïnstalleerd met:

```text
install_pokemonred_v3_fresh_zero.sh
```

De migratie:

- controleert eerst de hashes van de gereviewde broncode;
- compileert nieuwe files vóór installatie;
- stopt daarna pas bot/viewers;
- maakt een volledige `.tar.gz` backup met SHA-256;
- bewaart `persistent_agent_1.state` en metadata;
- hervat het laatste PPO-checkpoint;
- start 8080 en 6006 opnieuw;
- doet een runtime smoke-test;
- voert automatische rollback uit als de nieuwe architectuur niet start.

## RAM

Met de eerdere ~16 GB RAM-configuratie veroorzaakten 16 PPO agents `systemd-oomd` memory-pressure kills.

De huidige architectuur gebruikt:

```text
11 PPO trainingagents + 1 aparte read-only explorer
```

Bij hardwarewijzigingen opnieuw controleren:

```bash
free -h
cat /proc/pressure/memory
htop
iostat -xz 1
watch -n 1 nvidia-smi
```

## Snelle controle

```bash
cd ~/PokemonRedExperiments-v2/v2

echo "=== PPO ==="
pgrep -af baseline_fast_v2.py

echo "=== ARCHITECTUUR ==="
grep -E "PPO training envs|persistent explorer|loading checkpoint" pokemon_training.log

echo "=== TMUX ==="
tmux ls

echo "=== POORTEN ==="
ss -ltnp | grep -E ':8080|:6006'
```

Gewenste diensten:

```text
pokemonred
pokemonlive
tensorboard
```

## Statusnotitie

Deze README beschrijft de **Pokémon Red AI V3-architectuur** na succesvolle installatie met `install_pokemonred_v3_fresh_zero.sh`.

Na de V3-installatie staat Agent 1 buiten de PPO vector-environment en leveren alleen Agents 2–12 trainingsdata.
