# Pokémon Red AI V3 — uitgebreide technische README

## 1. Projectoverzicht

Deze installatie is **Pokémon Red AI V3**, een sterk aangepaste lokale voortzetting van:

**PWhiddy/PokemonRedExperiments**

```text
https://github.com/PWhiddy/PokemonRedExperiments
```

Upstream:

- auteur/maintainer: PWhiddy / Peter Whidden;
- projectdoel: reinforcement-learningagents Pokémon Red laten spelen;
- upstreamlicentie: MIT;
- belangrijke componenten: PyBoy, Stable-Baselines3 PPO, Gym/Gymnasium-achtige environments, `SubprocVecEnv` en TensorBoard.

Deze documentatie beschrijft de **lokale aangepaste installatie op de mining rig** en niet de officiële upstreamstatus.

**Naamgeving:** upstream gebruikt `v2` en de lokale directory/bestandsnamen zoals `v2/`, `baseline_fast_v2.py` en `red_gym_env_v2.py` blijven voorlopig behouden om bestaande imports en scripts niet te breken. De **lokale architectuur/release heet Pokémon Red AI V3**.

---

## 2. Belangrijkste wijziging van deze versie

De architectuur is gewijzigd zodat **Agent 1 geen trainingsdata meer aan PPO levert**.

De gewenste/nieuwe architectuur is:

```text
                            PPO MODEL
                               |
                   actuele gedeelde policy
                               |
          +--------------------+--------------------+
          |                                         |
          v                                         v
  PPO TRAININGSPAD                         READ-ONLY PAD
  Agents 2–12                             Agent 1
  11 environments                         persistent explorer
          |                                         |
          | observations                           | model.predict()
          | actions                                |
          | rewards                                v
          v                                 eigen Game Boy env
   PPO rollout buffer                       eigen wereldstate
          |                                  eigen flags/items
          |                                  eigen reward/heal
          v                                  eigen autosave
      PPO update                                     |
          |                                          X
          +------------------------------>  GEEN TERUGKOPPELING
                                             NAAR PPO BUFFER
```

Hiermee geldt:

> **Agent 1 kan PPO lezen, maar kan PPO niet trainen.**

---

## 3. Waarom Agent 1 uit PPO is gehaald

De eerdere lokale versie gebruikte 12 environments in dezelfde `SubprocVecEnv`.

Daardoor leverde ook de persistente Agent 1:

- observations;
- actions;
- rewards;
- transitions;

aan dezelfde PPO rollout-buffer.

Dat was ongewenst omdat Agent 1:

- een blijvende wereld heeft;
- andere rewardgeschiedenis kan hebben;
- veel verder kan doorlopen dan gewone episodes;
- daardoor een andere datadistributie heeft dan de reset-agents.

De nieuwe architectuur voorkomt dit volledig.

Agent 1 bestaat niet meer als PPO training-environment.

---

## 4. Exacte rol van Agent 1

Agent 1 is nu een aparte:

```text
[PERSISTENT READ-ONLY EXPLORER]
```

Hij doet:

1. een eigen `RedGymEnv` openen;
2. persistent world-state laden;
3. de actuele PPO-policy lezen;
4. `model.predict()` gebruiken om een actie te kiezen;
5. die actie in zijn eigen Pokémon-wereld uitvoeren;
6. viewer/debugmetadata schrijven;
7. zijn wereld periodiek autosaven;
8. doorgaan.

Hij doet **niet**:

- schrijven naar PPO rollout-buffer;
- PPO loss beïnvloeden;
- PPO rewardtraining beïnvloeden;
- gradients genereren;
- `model.learn()` data leveren.

### Belangrijk

De aparte explorer callback heeft wel toegang tot `self.model`, omdat hij tijdens de trainingsloop de actuele policy nodig heeft.

Dat is alleen **read access voor inference**.

---

## 5. Rol van Agents 2–12

Agents 2–12 zijn de enige echte PPO-trainingagents.

Aantal:

```text
11
```

Logische nummering:

```text
Agent 2
Agent 3
...
Agent 12
```

Technisch zijn dit de 11 environments in de `SubprocVecEnv`.

Hun transitions vullen:

```text
PPO rollout_buffer
```

en alleen hun ervaringen beïnvloeden:

- policy updates;
- value-function updates;
- PPO losses;
- advantages;
- gradients;
- trainingsreward.

---

## 6. PPO-timestepberekening

De huidige V3-configuratie gebruikt:

```text
n_steps = ep_length // 64
        = 163840 // 64
        = 2560
```

Met 11 PPO environments:

```text
2560 × 11 = 28160 PPO timesteps per rollout
```

Agent 1 telt hier bewust niet bij.

### Volledige gewone episodecyclus

Voor een gewone trainingagent:

```text
max_steps = 163840
```

Over 11 trainingsagents komt één volledige episodecyclus overeen met ongeveer:

```text
163840 × 11 = 1802240 globale PPO timesteps
```

---

## 7. PPO step counter bij checkpoint-resume

De nieuwe baseline gebruikt bij hervatten:

```python
model.learn(
    ...,
    reset_num_timesteps=False,
)
```

Dit betekent dat de zichtbare PPO `total_timesteps` vanaf het geladen checkpoint hoort verder te tellen.

Dit is **niet** een instelling voor Agent 1.

Het hoort bij:

```text
PPO model
+
Agents 2–12 trainingdata
```

Agent 1 heeft zijn eigen `step_count` in de persistent explorer.

Dus:

```text
PPO steps
!=
Agent 1 steps
```

---

## 8. Agent 1 volledig continu

De nieuwe gewenste versie laat Agent 1 niet meer resetten op de gewone `max_steps`.

Voor Agent 1:

```python
if self.persistent_agent:
    return False
```

in de normale `check_if_done()`-logica.

Daarom blijven tijdens het draaiende explorerproces onder andere doorgaan:

- `step_count`;
- heal/rewardadministratie;
- exploration state;
- `reward_sum`;
- zijn Game Boy-wereld;
- items;
- flags;
- locatie.

Dit is mogelijk omdat Agent 1 niet meer in de PPO-training zit en zijn cumulatieve episodekarakter daardoor de PPO-distributie niet meer kan vervormen.

### Belangrijke beperking

De **Game Boy-worldstate** wordt persistent op disk bewaard.

Python-only counters worden momenteel niet allemaal als volledige countersnapshot bewaard.

Dus bij een volledige explorer/Python-procesherstart:

- Pokémon-wereld: behouden;
- positie/items/flags: via save-state behouden;
- sommige Python-counters: kunnen vanaf nieuwe proceswaarden verdergaan/resetten afhankelijk van wat expliciet persistent is gemaakt.

Binnen één draaiend Agent-1-proces worden ze niet meer periodiek door `max_steps` gereset.

---

## 9. Persistent world-state

Bestand:

```text
runs/persistent_agent_1.state
```

Doel:

- Game Boy wereld bewaren;
- na procesrestart terugkomen in dezelfde wereld;
- niet opnieuw volledig vanuit `init.state` hoeven starten.

Autosave blijft actief.

De eerdere lokale configuratie gebruikte ongeveer:

```text
5000 eigen Agent-1 steps
```

per periodieke autosave.

---

## 10. Persistent Last flag / metadata

Metadata:

```text
runs/persistent_agent_1_meta.json
```

Hierin kan viewer/debuginformatie staan zoals:

```text
last_flag
seen_flag_keys
agent_number
step
```

### Zeer belangrijk

Dit bestand is **geen PPO-geheugen**.

Een verkeerde `Last flag`-naam:

```text
kan viewerlabel fout maken
```

maar maakt niet automatisch fout:

```text
PPO gewichten
Game Boy event bits
PPO rollout data
```

Bovendien is Agent 1 nu read-only voor PPO, dus zijn viewerflagmetadata kan de PPO-training helemaal niet beïnvloeden.

---

## 11. Event flags

### LSB-first fix

De lokale eventdecoder werd aangepast van een foutgevoelige binary-stringinterpretatie naar:

```python
for idx in range(8):
    if val & (1 << idx):
        ...
```

Hiermee worden bitnummers uit `events.json` correct LSB-first geïnterpreteerd.

### Flags-teller

`Flags: 9` betekent:

```text
9 actuele relevante event bits voor die agent
```

Het betekent niet:

```text
"verhaalflag nummer 9"
```

### Last flag

`Last flag` is een leesbare voortgangs/debugwaarde.

Hij wordt niet gebruikt als PPO-observation of trainingstarget.

---

## 12. TensorBoard flagsemantiek

De gewenste huidige semantiek:

### `trajectory/all_flags`

Alle benoemde flags die in **die specifieke voltooide PPO-episode** bereikt werden.

Een volgende episode mag dus lager zijn:

```text
episode A: 15 flags
episode B: 9 flags
episode C: 13 flags
```

Dat is juist en nuttig.

### `trajectory/current_flags`

Zelfde episodecontext onder een explicietere naam.

### `trajectory/ever_discovered_flags`

Aparte cumulatieve historiek.

Dit mag alleen oplopen.

Het mag niet worden geïnterpreteerd als actuele episodeprogressie.

### Agent 1

De TensorBoard PPO-episodeflags horen voortaan bij de **11 trainingagents**, niet bij Agent 1.

Agent 1 is read-only en zit niet in de PPO vector environment.

---

## 13. TensorBoard callback na de architectuurwijziging

Voorheen werd speciale logica gebruikt omdat Agent 1 onderdeel was van de vector env en afwijkend resetgedrag had.

Na de read-only splitsing:

```text
training env index 0 = logische Agent 2
```

TensorBoard kan dus weer de eerste echte PPO-environment gebruiken voor de episode-trigger.

Agent 1 staat volledig buiten dit trainingspad.

---

## 14. Projectpaden

Projectroot:

```text
/home/nico/PokemonRedExperiments-v2
```

V2:

```text
/home/nico/PokemonRedExperiments-v2/v2
```

ROM:

```text
/home/nico/PokemonRedExperiments-v2/PokemonRed.gb
```

Init-state:

```text
/home/nico/PokemonRedExperiments-v2/init.state
```

Runs:

```text
/home/nico/PokemonRedExperiments-v2/v2/runs
```

Traininglog:

```text
/home/nico/PokemonRedExperiments-v2/v2/pokemon_training.log
```

Conda:

```text
pokemonredv2
```

---

## 15. Hardware / software tijdens ontwikkeling

Gebruikte setup:

- Ubuntu Linux;
- NVIDIA GeForce RTX 3060 Ti;
- PyTorch `2.5.0+cu124`;
- CUDA runtime via PyTorch: `12.4`;
- Python 3.11;
- Stable-Baselines3 PPO;
- PyBoy;
- TensorBoard.

GPU-test:

```bash
conda activate pokemonredv2

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
PY
```

---

## 16. ROM

Verwachte lokale ROM:

```text
/home/nico/PokemonRedExperiments-v2/PokemonRed.gb
```

Verwachte SHA-1:

```text
ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

Controle:

```bash
sha1sum ~/PokemonRedExperiments-v2/PokemonRed.gb
```

De ROM zelf wordt niet met de README/projectbundel gedeeld.

---

## 17. Nieuwe hoofdsourcefiles

Na succesvolle migratie zijn vooral deze bestanden belangrijk:

```text
baseline_fast_v2.py
red_gym_env_v2.py
persistent_explorer_callback.py
tensorboard_callback.py
live_server.py
```

### `baseline_fast_v2.py`

Verantwoordelijk voor:

- 11 PPO training envs;
- logische nummering Agents 2–12;
- PPO model;
- checkpoint loading;
- `reset_num_timesteps=False`;
- callbacks;
- koppeling van de read-only explorer callback.

### `persistent_explorer_callback.py`

Nieuw bestand.

Verantwoordelijk voor:

- Agent 1 zijn aparte env;
- persistent world laden;
- per callbackstep policy lezen;
- `model.predict()` uitvoeren;
- Agent 1 env stappen;
- géén rollout-bufferinteractie.

### `red_gym_env_v2.py`

Bevat onder andere:

- gameenvironment;
- rewards;
- flags;
- viewer metadata;
- persistent state;
- Agent 1 no-max-step-reset.

### `tensorboard_callback.py`

Logt PPO-data van de trainingagents.

### `live_server.py`

Toont alle logische agents, met Agent 1 apart gelabeld als read-only persistent explorer.

---

## 18. Migratiescript

Gebruik voor deze architectuur:

```text
install_pokemonred_v3_fresh_zero.sh
```

**Gebruik de eerdere `migrate_agent1_readonly_ppo.sh` niet voor deze gewenste variant.**

Het nieuwe script is gebouwd op basis van de op 2026-08-19 geüploade en gereviewde bronbundle.

---

## 19. Wat de migratie eerst controleert

Vóór de bot wordt gestopt controleert het script SHA-256 hashes van de gereviewde huidige bestanden.

Hierdoor wordt niet blind op onverwacht veranderde source gepatcht.

Als een bestand afwijkt:

```text
migratie stopt
bot blijft draaien
niets wordt gewijzigd
```

---

## 20. Offline syntax- en architectuurcheck

De migratie schrijft de nieuwe files eerst naar een tijdelijke map.

Daar gebeurt:

```text
py_compile
+
statische architectuurchecks
```

Onder andere wordt gecontroleerd:

- exact 11 PPO envs;
- training envs zijn logische Agents 2–12;
- geen training env heeft `persistent_agent=True`;
- Agent 1 callback is gekoppeld;
- callback gebruikt `model.predict()`;
- callback bevat geen `rollout_buffer`;
- Agent 1 env is persistent;
- Agent 1 normale `max_steps` done staat uit;
- TensorBoard gebruikt de eerste echte PPO env.

Pas daarna stopt het script de huidige bot.

---

## 21. Backup vóór migratie

Na de bronchecks en nadat de oude bot netjes gestopt is, maakt de migratie een volledige backup:

```text
~/pokemonred_backups/
PokemonRedExperiments-v2_before_agent1_readonly_continuous_<timestamp>.tar.gz
```

Plus:

```text
.tar.gz.sha256
```

Daarna:

```bash
sha256sum -c <backup>.sha256
```

moet `OK` geven.

Er wordt daarnaast een codebackup gemaakt in:

```text
v2/readonly_agent1_backups/<timestamp>/
```

---

## 22. Automatische rollback

Als na installatie:

- PPO niet start;
- `Traceback` verschijnt;
- checkpoint niet wordt geladen;
- read-only explorer niet initialiseert;
- 8080 niet luistert;
- 6006 niet luistert;

probeert het migratiescript automatisch de oude sourcefiles terug te zetten en de oude architectuur vanaf hetzelfde checkpoint opnieuw te starten.

De volledige `.tar.gz` backup blijft daarnaast bestaan.

---

## 23. Checkpoints

Checkpointbestanden:

```text
runs/poke_<timesteps>_steps.zip
```

Laatste checkpoint:

```bash
cd ~/PokemonRedExperiments-v2/v2

LATEST=$(find runs -maxdepth 1 -type f -name 'poke_*_steps.zip' | sort -V | tail -n 1)
echo "$LATEST"
```

Hervatten:

```bash
CHECKPOINT="${LATEST%.zip}"
```

De baseline leest de checkpointnaam via stdin.

Controle:

```bash
grep -i "loading checkpoint" pokemon_training.log
```

---

## 24. Checkpointfrequentie met 11 PPO envs

De bestaande callback gebruikt:

```text
save_freq = ep_length // 2
          = 81920 callback calls
```

Met 11 vectorized training envs komt dit neer op ongeveer:

```text
81920 × 11 = 901120 globale PPO timesteps
```

per checkpointinterval.

Agent 1 telt hier niet in mee.

---

## 25. Training starten vanaf nul

```bash
cd ~/PokemonRedExperiments-v2/v2

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; python -u baseline_fast_v2.py 2>&1 | tee pokemon_training.log; echo; echo TRAINING GESTOPT; exec bash'"
```

Bij een nieuwe training:

- PPO start vanaf nieuwe gewichten;
- Agents 2–12 trainen;
- Agent 1 leest de nieuwe policy;
- als `persistent_agent_1.state` nog bestaat, kan Agent 1 zijn bestaande wereld blijven gebruiken.

Als een volledig nieuwe Agent-1 wereld gewenst is, moet zijn persistent state bewust apart worden verwijderd of gearchiveerd.

---

## 26. Hervatten vanaf laatste checkpoint

```bash
cd ~/PokemonRedExperiments-v2/v2

LATEST=$(find runs -maxdepth 1 -type f -name 'poke_*_steps.zip' | sort -V | tail -n 1)
CHECKPOINT="${LATEST%.zip}"

tmux new-session -d -s pokemonred \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; echo \"$CHECKPOINT\" | python -u baseline_fast_v2.py 2>&1 | tee pokemon_training.log; echo; echo TRAINING GESTOPT; exec bash'"
```

Na de nieuwe architectuur hoort de PPO teller door te lopen door:

```python
reset_num_timesteps=False
```

Controle:

```bash
grep -i "loading checkpoint" pokemon_training.log
```

---

## 27. Bot netjes stoppen

```bash
cd ~/PokemonRedExperiments-v2/v2

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

---

## 28. Live viewer — 8080

Start:

```bash
cd ~/PokemonRedExperiments-v2/v2

tmux kill-session -t pokemonlive 2>/dev/null || true

tmux new-session -d -s pokemonlive \
"cd /home/nico/PokemonRedExperiments-v2/v2 && /home/nico/miniconda3/envs/pokemonredv2/bin/python -u live_server.py"
```

Controle:

```bash
ss -ltnp | grep :8080
```

Browser:

```text
http://192.168.129.83:8080
```

De viewer sorteert op logisch agentnummer.

Agent 1 hoort als eerste te staan en read-only/persistent gelabeld te worden.

---

## 29. TensorBoard — 6006

Start:

```bash
cd ~/PokemonRedExperiments-v2/v2

tmux kill-session -t tensorboard 2>/dev/null || true

tmux new-session -d -s tensorboard \
"bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate pokemonredv2; cd ~/PokemonRedExperiments-v2/v2; tensorboard --logdir runs --host 0.0.0.0 --port 6006'"
```

Controle:

```bash
ss -ltnp | grep :6006
```

Browser:

```text
http://192.168.129.83:6006
```

---

## 30. Live stats interpreteren

### Globale FPS

Globale trainingssnelheid van PPO / training loop.

### PPO steps

`total_timesteps` van het PPO-model.

Na de read-only split gaat dit over **11 PPO training environments**.

Agent 1 hoort hier niet bij.

### Agent 1 Steps

Eigen step counter van de read-only explorer.

### Flags

Actuele event bits van die specifieke Pokémon-wereld.

### Last flag

Viewer/debugmetadata.

### Heal / explore / reward sum

Voor Agent 1 mogen deze tijdens het draaiende explorerproces blijven doorlopen.

Ze trainen PPO niet.

---

## 31. Belangrijk verschil: modelgeheugen vs wereldgeheugen

### PPO modelgeheugen

```text
checkpoint .zip
```

Wordt alleen geleerd uit Agents 2–12.

### Agent 1 wereldgeheugen

```text
persistent_agent_1.state
```

Is zijn eigen Game Boy wereld.

### Viewer metadata

```text
persistent_agent_1_meta.json
```

Is debug/loginformatie.

Dit zijn drie aparte concepten.

---

## 32. Waarom Agent 1 PPO niet meer kan beïnvloeden

PPO krijgt zijn rollout uitsluitend van:

```python
SubprocVecEnv([
    make_env(... Agent 2 ...),
    ...
    make_env(... Agent 12 ...),
])
```

Agent 1 wordt apart vanuit de callback gestapt.

Zijn actiepad is:

```text
current PPO model
    ↓
model.predict()
    ↓
Agent 1 env.step()
    ↓
eigen wereld
```

Er is geen:

```text
Agent 1 env transition
    ↓
rollout_buffer.add(...)
```

in zijn callback.

Daarom wordt zijn data niet gebruikt in de PPO-update.

---

## 33. Hoe Agent 1 toch "slimmer" wordt

De policy wordt voortdurend verbeterd door Agents 2–12.

Agent 1 gebruikt bij latere callbacksteps dezelfde actuele modelobject/policy.

Dus:

```text
Agents 2–12 ontdekken iets
        ↓
PPO update
        ↓
policy verandert
        ↓
Agent 1 gebruikt daarna de nieuwere policy
```

Agent 1 ontvangt dus kennis **éénrichtingsverkeer**:

```text
PPO -> Agent 1
```

niet:

```text
Agent 1 -> PPO
```

---

## 34. Trainingfase waarin agents stil lijken te staan

Tijdens PPO updates kan de emulatorweergave tijdelijk stilstaan.

Dat is normaal:

```text
rollout verzamelen
    ↓
PPO neural-network update
    ↓
volgende rollout
```

Als:

- `iterations` stijgt;
- `total_timesteps` stijgt;
- proces actief blijft;

is er geen echte freeze.

---

## 35. Traininglog controleren

```bash
tail -30 ~/PokemonRedExperiments-v2/v2/pokemon_training.log
```

Zoek:

```text
fps
iterations
total_timesteps
```

Architectuurmeldingen:

```bash
grep -E \
'PPO training envs|read-only persistent explorer|loading checkpoint' \
~/PokemonRedExperiments-v2/v2/pokemon_training.log
```

---

## 36. Tmux

```bash
tmux ls
```

Gewenst:

```text
pokemonred
pokemonlive
tensorboard
```

Attach:

```bash
tmux attach -t pokemonred
```

Detach:

```text
Ctrl+B
D
```

---

## 37. Poorten

```bash
ss -ltnp | grep -E ':8080|:6006'
```

Gewenst:

```text
0.0.0.0:8080
0.0.0.0:6006
```

---

## 38. Disk-I/O optimalisaties

Tijdens eerdere experimenten liep `/dev/sda` bijna volledig vol qua I/O-utilisatie.

Aanpassingen:

- `print_rewards=False`;
- minder vaak live JPEG schrijven;
- één zichtbaar agentscherm tegelijk;
- refreshfrequentie beperkt;
- oude frames wissen na restart.

Dit verminderde I/O-pressure.

---

## 39. RAM / systemd-oomd geschiedenis

Met 16 parallelle PPO envs en ongeveer 16 GB RAM trad memory pressure op.

`systemd-oomd` heeft toen volledige tmux scopes beëindigd.

Daarom werd later met 12 envs gewerkt.

De nieuwe split gebruikt:

```text
11 PPO envs
+
1 aparte read-only explorer
```

Dit blijft 12 Game Boy-environments, maar slechts 11 zitten in de PPO vector environment.

Controleer:

```bash
free -h
cat /proc/pressure/memory
htop
```

---

## 40. GPU

PyBoy/environmentwerk is grotendeels CPU-side.

PPO neural-networkwerk gebruikt de GPU wanneer CUDA beschikbaar is.

Controle:

```bash
watch -n 1 nvidia-smi
```

Een momentopname van 0% GPU betekent niet automatisch dat de training CPU-only is.

---

## 41. Bestaande backups

Eerdere volledige backup:

```text
/home/nico/pokemonred_backups/PokemonRedExperiments-v2_before_persistent_20260819_092227.tar.gz
```

De SHA-256-verificatie daarvan gaf `OK`.

Nieuwe migraties maken daarnaast een backup zoals:

```text
PokemonRedExperiments-v2_before_agent1_readonly_continuous_<timestamp>.tar.gz
```

met `.sha256`.

---

## 42. Backup terugzetten

Eerst stoppen:

```bash
tmux kill-session -t pokemonred 2>/dev/null || true
tmux kill-session -t pokemonlive 2>/dev/null || true
tmux kill-session -t tensorboard 2>/dev/null || true
pkill -f '[b]aseline_fast_v2.py' 2>/dev/null || true
```

Huidige map archiveren:

```bash
cd ~
mv PokemonRedExperiments-v2 PokemonRedExperiments-v2_broken
```

Backup terugzetten:

```bash
tar -xzf ~/pokemonred_backups/<backup>.tar.gz -C ~
```

Verifiëren:

```bash
sha256sum -c ~/pokemonred_backups/<backup>.tar.gz.sha256
```

---

## 43. Persistent bestanden die je niet zomaar moet wissen

Als Agent 1 zijn wereld moet behouden:

```text
runs/persistent_agent_1.state
runs/persistent_agent_1_meta.json
```

Bij gewone live-cache cleanup alleen:

```bash
rm -f runs/curframe_*.jpeg
rm -f runs/curframe_*.json
```

---

## 44. Een volledig nieuwe run starten

Er zijn twee verschillende dingen die je kunt resetten.

### Alleen PPO opnieuw vanaf nul

- checkpoints niet laden;
- nieuwe PPO modelgewichten;
- Agent 1 state kan eventueel behouden blijven.

### Alles werkelijk volledig opnieuw

Ook:

```text
persistent_agent_1.state
persistent_agent_1_meta.json
```

bewust archiveren/verwijderen.

Doe dit alleen als je ook Agent 1 zijn wereld wilt wissen.

---

## 45. Bekende beperking: Agent 1 Python counters over procesrestart

De nieuwe variant voorkomt max-step resets voor Agent 1.

Daardoor lopen binnen het proces door:

```text
steps
heal
explore
reward sum
```

Maar een Game Boy save-state bevat niet automatisch alle Python-side environmentvariabelen.

Daarom geldt momenteel:

```text
soft/no max-step reset -> counters blijven lopen
volledige Python restart -> world blijft, sommige counters kunnen resetten
```

Dit is bewust gedocumenteerd zodat `persistent world` niet verward wordt met `volledige Python state persistence`.

---

## 46. Last flag bekende aandachtspunten

`Last flag` is logging.

Bij historische wijzigingen aan de tracker kon een oude naam zichtbaar blijven.

Dit heeft geen invloed op PPO.

Na read-only scheiding kan Agent 1 zijn Last flag sowieso niet meer aan PPO doorgeven, omdat Agent 1 geen PPO trainingdata produceert.

---

## 47. Oude TensorBoard runs

Codewijzigingen veranderen bestaande eventfiles niet retroactief.

Een oude `poke_ppo_X` kan dus oudere semantiek tonen.

Voor de juiste interpretatie altijd kijken naar de run die ná de relevante codefix is aangemaakt.

---

## 48. Migratiecheck na installatie

Na succesvolle `install_pokemonred_v3_fresh_zero.sh`:

```bash
cd ~/PokemonRedExperiments-v2/v2

grep -E \
'PPO training envs|read-only persistent explorer|loading checkpoint' \
pokemon_training.log
```

Verwacht conceptueel:

```text
PPO training envs = 11 (logical Agents 2-12)
Agent 1 = read-only persistent explorer
loading checkpoint
```

Controleer ook:

```bash
pgrep -af baseline_fast_v2.py
tmux ls
ss -ltnp | grep -E ':8080|:6006'
```

---

## 49. Rollout-buffer veiligheidscontrole

De aparte explorerfile:

```text
persistent_explorer_callback.py
```

hoort géén code te bevatten die Agent 1 in:

```text
rollout_buffer
```

schrijft.

Snelle controle:

```bash
grep -n "rollout_buffer" persistent_explorer_callback.py
```

Gewenst:

```text
geen output
```

Policy inference:

```bash
grep -n "model.predict" persistent_explorer_callback.py
```

Daar hoort wel een match te zijn.

---

## 50. Samenvatting lokale wijzigingen

Historisch en huidig:

1. upstream V2 gekloond;
2. Python/Conda/CUDA opgezet;
3. RTX 3060 Ti gevalideerd;
4. worker count aangepast;
5. live frames geactiveerd;
6. reward console spam uitgezet;
7. 8080 live viewer gemaakt;
8. 6006 TensorBoard toegevoegd;
9. viewer naar één agent tegelijk geoptimaliseerd;
10. live JSON metadata toegevoegd;
11. globale PPO steps toegevoegd;
12. eventbitvolgorde LSB-first gecorrigeerd;
13. chronologische Last flag tracking toegevoegd;
14. TensorBoard `all_flags` per episode hersteld;
15. aparte cumulatieve flaghistoriek toegevoegd;
16. Agent 1 worldstate persistent gemaakt;
17. autosave toegevoegd;
18. soft-resetexperiment uitgevoerd;
19. persistent Last flag metadata toegevoegd;
20. RAM/OOM-problemen met 16 agents onderzocht;
21. stabielere lagere workerconfig gebruikt;
22. volledige backups + checksumprocedures toegevoegd;
23. **Agent 1 uit PPO rollout/training gehaald**;
24. **Agents 2–12 als 11 enige PPO-trainingagents ingesteld**;
25. **read-only persistent explorer callback toegevoegd**;
26. **Agent 1 leest de policy via `model.predict()`**;
27. **Agent 1 max-step reset verwijderd**;
28. **Agent 1 heal/explore/reward/steps mogen binnen zijn proces doorlopen**;
29. **PPO checkpoint-resume gebruikt `reset_num_timesteps=False`**;
30. **migratie heeft bronhashchecks, precompile, backup, smoke-test en rollback**.

---

## 51. Upstream credits

Originele repository:

```text
https://github.com/PWhiddy/PokemonRedExperiments
```

Upstream auteur/maintainer:

```text
PWhiddy / Peter Whidden
```

Project:

```text
Train RL agents to play Pokemon Red
```

Licentie:

```text
MIT
```

Deze lokale README documenteert de aangepaste experimentele architectuur en moet niet als officiële upstreamdocumentatie worden beschouwd.

---

## 52. Documentstatus

Datum:

```text
2026-08-19
```

Deze README beschrijft de **Pokémon Red AI V3 read-only continuous Agent 1-architectuur na succesvolle installatie met**:

```text
install_pokemonred_v3_fresh_zero.sh
```

Als dat script nog niet succesvol is uitgevoerd, kan de daadwerkelijk draaiende code nog de eerdere 12-env architectuur gebruiken.

Aanbevolen werkwijze voor verdere wijzigingen:

1. huidige codebundle bewaren;
2. volledige backup maken;
3. één architectuurwijziging tegelijk;
4. `py_compile` uitvoeren;
5. bot gecontroleerd stoppen;
6. checkpoint veilig hervatten;
7. 8080/6006 controleren;
8. `pokemon_training.log` inspecteren;
9. memory pressure controleren;
10. pas daarna verder wijzigen.
