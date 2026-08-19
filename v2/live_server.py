from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json
import urllib.parse
import re

BASE = Path(__file__).resolve().parent
ROOT = BASE / "runs"
LOG_FILE = BASE / "pokemon_training.log"

HTML = r"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pokemon Red AI - Live Agent</title>

<style>
body {
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    margin: 16px;
}

h1 {
    text-align: center;
    margin: 8px 0 8px;
}

#status {
    text-align: center;
    margin-bottom: 16px;
    font-size: 18px;
}

#controls {
    text-align: center;
    margin-bottom: 16px;
}

button {
    font-size: 17px;
    padding: 8px 18px;
    margin: 0 8px;
    cursor: pointer;
}

#counter {
    display: inline-block;
    min-width: 150px;
    font-size: 18px;
}

#agent {
    width: 360px;
    margin: 0 auto;
    background: #222;
    padding: 14px;
    border-radius: 10px;
    box-sizing: border-box;
}

#name {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Iets groter dan origineel, maar niet schermvullend */
#screen {
    display: block;
    width: 320px;
    height: auto;
    margin: 0 auto;
    background: black;
    image-rendering: pixelated;
}

#meta {
    margin-top: 12px;
    background: #181818;
    border-radius: 6px;
    padding: 11px;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.5;
    white-space: pre-wrap;
    text-align: left;
    box-sizing: border-box;
}
</style>
</head>

<body>

<h1>Pokemon Red AI - Live Agent</h1>

<div id="status">Agents zoeken...</div>

<div id="controls">
    <button onclick="previousAgent()">◀ Vorige</button>
    <span id="counter">-</span>
    <button onclick="nextAgent()">Volgende ▶</button>
</div>

<div id="agent">
    <div id="name">Agent</div>
    <img id="screen">
    <div id="meta">Metadata laden...</div>
</div>

<script>
let items = [];
let current = 0;
let globalFps = null;
let totalPpoSteps = null;

function formatNumber(value) {
    if (value === undefined || value === null) {
        return "0.00";
    }
    return Number(value).toFixed(2);
}

function formatInt(value) {
    if (value === undefined || value === null) {
        return "-";
    }
    return Number(value).toLocaleString("nl-BE");
}

function formatMeta(meta) {
    if (!meta) {
        return [
            "PPO FPS     : " + (globalFps ?? "-"),
            "PPO steps   : " + formatInt(totalPpoSteps),
            "",
            "Metadata wordt opgebouwd..."
        ].join("\n");
    }

    return [
        "PPO FPS     : " + (globalFps ?? "-"),
        "Steps       : " + formatInt(meta.step),
        "PPO steps   : " + formatInt(totalPpoSteps),
        "Flags       : " + (meta.flags ?? "-"),
        "Last flag   : " + (meta.last_flag ?? "-"),
        "Badges      : " + (meta.badges ?? "-"),
        "Event       : " + formatNumber(meta.event),
        "Heal        : " + formatNumber(meta.heal),
        "Badge rew.  : " + formatNumber(meta.badge_reward),
        "Explore     : " + formatNumber(meta.explore),
        "Stuck       : " + formatNumber(meta.stuck),
        "Reward sum  : " + formatNumber(meta.reward_sum),
        "Done        : " + String(meta.done ?? false)
    ].join("\n");
}

async function loadData() {
    try {
        const response = await fetch("/frames?t=" + Date.now());
        const data = await response.json();

        items = data.items || [];
        globalFps = data.global_fps;
        totalPpoSteps = data.total_ppo_steps;

        if (current >= items.length) {
            current = Math.max(0, items.length - 1);
        }

        document.getElementById("status").textContent =
            "Actieve agents: " + items.length +
            " | PPO-train agents: 11 | Explorer: 1 READ-ONLY" +
            " | PPO FPS: " + (globalFps ?? "-") +
            " | PPO steps: " + formatInt(totalPpoSteps);

        drawAgent();
    } catch (error) {
        document.getElementById("status").textContent =
            "Viewer fout: " + error;
    }
}

function drawAgent() {
    if (!items.length) {
        document.getElementById("counter").textContent = "Geen agents";
        document.getElementById("name").textContent = "Geen live agent";
        document.getElementById("screen").removeAttribute("src");
        document.getElementById("meta").textContent = "Geen metadata";
        return;
    }

    const item = items[current];

    const shownAgentNumber =
        (item.meta && item.meta.agent_number) ? item.meta.agent_number : (current + 1);

    document.getElementById("name").textContent =
        "Agent " + shownAgentNumber +
        ((item.meta && item.meta.persistent) ? " [PERSISTENT READ-ONLY]" : "");

    document.getElementById("counter").textContent =
        (current + 1) + " / " + items.length;

    document.getElementById("meta").textContent =
        formatMeta(item.meta);

    refreshImage();
}

function refreshImage() {
    if (!items.length) {
        return;
    }

    const item = items[current];
    document.getElementById("screen").src =
        "/" + item.image + "?t=" + Date.now();
}

function nextAgent() {
    if (!items.length) {
        return;
    }

    current++;

    if (current >= items.length) {
        current = 0;
    }

    drawAgent();
}

function previousAgent() {
    if (!items.length) {
        return;
    }

    current--;

    if (current < 0) {
        current = items.length - 1;
    }

    drawAgent();
}

document.addEventListener("keydown", function(event) {
    if (event.key === "ArrowRight") {
        nextAgent();
    }

    if (event.key === "ArrowLeft") {
        previousAgent();
    }
});

/* Alleen het huidige live beeld vernieuwen */
setInterval(refreshImage, 1500);

/* Stats en PPO-info vernieuwen */
setInterval(loadData, 5000);

loadData();
</script>

</body>
</html>
"""

def parse_training_stats():
    result = {
        "fps": None,
        "total_timesteps": None,
    }

    if not LOG_FILE.exists():
        return result

    try:
        text = LOG_FILE.read_text(errors="ignore")

        fps_values = re.findall(
            r"\|\s+fps\s+\|\s+(\d+)",
            text
        )
        if fps_values:
            result["fps"] = int(fps_values[-1])

        step_values = re.findall(
            r"\|\s+total_timesteps\s+\|\s+(\d+)",
            text
        )
        if step_values:
            result["total_timesteps"] = int(step_values[-1])

    except Exception:
        pass

    return result

def read_meta(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(ROOT),
            **kwargs,
        )

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path

        if path == "/":
            data = HTML.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )
            self.send_header(
                "Content-Length",
                str(len(data))
            )
            self.send_header(
                "Cache-Control",
                "no-store"
            )
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/frames":
            items = []

            for img in sorted(ROOT.glob("curframe_*.jpeg")):
                items.append({
                    "image": img.name,
                    "meta": read_meta(img.with_suffix(".json")),
                })

            items.sort(
                key=lambda item: int((item.get("meta") or {}).get("agent_number", 999))
            )

            training = parse_training_stats()

            payload = {
                "global_fps": training["fps"],
                "total_ppo_steps": training["total_timesteps"],
                "items": items,
            }

            data = json.dumps(payload).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.send_header(
                "Content-Length",
                str(len(data))
            )
            self.send_header(
                "Cache-Control",
                "no-store"
            )
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()

    def end_headers(self):
        self.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate"
        )
        super().end_headers()

print("Pokemon Red AI live viewer")
print("1 agent tegelijk")
print("Groter live beeld + Last flag + PPO steps")
print("Poort 8080")

ThreadingHTTPServer(
    ("0.0.0.0", 8080),
    Handler,
).serve_forever()
