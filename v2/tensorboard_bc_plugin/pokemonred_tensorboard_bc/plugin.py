from __future__ import annotations

import json
import os
from pathlib import Path

from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from werkzeug import wrappers


_JS = r"""
function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtStep(v) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n.toLocaleString("en-US") : String(v ?? "");
}

function routeText(route) {
  if (!Array.isArray(route) || !route.length) return "—";
  return route.map(x => esc(x)).join(" → ");
}

function targetRows(bc) {
  const targets = bc && bc.targets ? bc.targets : {};
  const entries = Object.entries(targets);
  if (!entries.length) return '<div class="empty">Nog geen BC targets.</div>';

  entries.sort((a, b) => {
    const am = Number(a[1]?.mastery_factor || 0);
    const bm = Number(b[1]?.mastery_factor || 0);
    if ((am > 0) !== (bm > 0)) return am > 0 ? -1 : 1;
    return String(a[0]).localeCompare(String(b[0]));
  });

  return entries.map(([name, r], i) => {
    const mastery = Number(r?.mastery_factor || 0);
    const agents = Number(
      r?.successful_agent_count ??
      (Array.isArray(r?.successful_agents) ? r.successful_agents.length : 0)
    );
    return `
      <div class="flagrow">
        <div class="flagtitle">
          <span class="flagnum">Flag ${i + 1}</span>
          <span class="flagname">${esc(name)}</span>
          <span class="state">${mastery > 0 ? "ACTIVE" : "MASTERED"}</span>
        </div>
        <div class="route"><b>Route:</b> ${routeText(r?.route)}</div>
        <div class="meta">
          Agents: ${agents} &nbsp; • &nbsp;
          Mastery: ${Math.round(mastery * 100)}%
        </div>
      </div>`;
  }).join("");
}

function snapshotCard(item, live=false) {
  const bc = live ? item : (item?.bc || {});
  const title = live ? "LIVE BC" : `step ${fmtStep(item?.step)}`;
  const run = live ? "" : (item?.run || "");
  const subtitle = live
    ? `BC version ${esc(bc?.version ?? 0)}`
    : `${esc(run)}${run ? " • " : ""}BC version ${esc(bc?.version ?? 0)}`;

  return `
    <section class="card ${live ? "live" : ""}">
      <div class="cardhead">
        <div>
          <div class="steptitle">${title}</div>
          <div class="sub">${subtitle}</div>
        </div>
        <div class="stats">
          <span>Targets <b>${esc(bc?.total_targets ?? 0)}</b></span>
          <span>Active <b>${esc(bc?.active_targets ?? 0)}</b></span>
          <span>Mastered <b>${esc(bc?.mastered_targets ?? 0)}</b></span>
          <span>Discoveries <b>${esc(bc?.total_discoveries ?? 0)}</b></span>
        </div>
      </div>
      ${targetRows(bc)}
    </section>`;
}

export function render() {
  document.body.innerHTML = `
    <style>
      :root { color-scheme: dark; }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: #212121;
        color: #f2f2f2;
      }
      .wrap { padding: 18px 22px 60px; }
      .topline {
        display:flex; align-items:center; justify-content:space-between;
        gap:16px; margin-bottom:14px;
      }
      h1 { font-size:22px; margin:0; font-weight:600; }
      .hint { opacity:.7; font-size:13px; }
      .card {
        border:1px solid #555; border-radius:4px; margin:0 0 16px;
        background:#292929; overflow:hidden;
      }
      .card.live { border-width:2px; }
      .cardhead {
        display:flex; justify-content:space-between; gap:20px;
        padding:13px 15px; border-bottom:1px solid #4a4a4a;
      }
      .steptitle { font-size:17px; font-weight:700; }
      .sub { opacity:.65; font-size:12px; margin-top:3px; }
      .stats { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; }
      .flagrow { padding:11px 15px; border-bottom:1px solid #3b3b3b; }
      .flagrow:last-child { border-bottom:0; }
      .flagtitle { display:flex; flex-wrap:wrap; align-items:center; gap:9px; }
      .flagnum { font-weight:700; min-width:54px; }
      .flagname { font-weight:600; }
      .state {
        font-size:10px; font-weight:700; letter-spacing:.5px;
        border:1px solid #777; border-radius:10px; padding:2px 7px;
      }
      .route {
        margin-top:7px;
        font-family:Consolas, "Courier New", monospace;
        line-height:1.45; overflow-wrap:anywhere;
      }
      .meta { margin-top:5px; opacity:.68; font-size:12px; }
      .empty { padding:15px; opacity:.65; }
      .error { border:1px solid #777; padding:15px; border-radius:4px; }
      @media (max-width:800px) { .cardhead { flex-direction:column; } }
    </style>
    <div class="wrap">
      <div class="topline">
        <h1>Breadcrumb Memory</h1>
        <div class="hint">Live refresh om de 5 seconden</div>
      </div>
      <div id="content">BC laden…</div>
    </div>`;

  const content = document.getElementById("content");
  const dataUrl = new URL("data", import.meta.url);

  async function refresh() {
    try {
      const res = await fetch(dataUrl, {cache: "no-store"});
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      let html = "";
      if (data.current) {
        html += snapshotCard(data.current, true);
      } else {
        html += '<div class="error">Live BC snapshot niet beschikbaar.</div>';
      }

      const history = Array.isArray(data.history) ? data.history : [];
      if (history.length) {
        html += history.slice().reverse().map(x => snapshotCard(x, false)).join("");
      } else {
        html += `
          <div class="empty">
            Nog geen historische BC snapshots. Vanaf de volgende nieuwe
            trajectory/all_flags/text_summary wordt BC op dezelfde step opgeslagen.
          </div>`;
      }
      content.innerHTML = html;
    } catch (e) {
      content.innerHTML =
        `<div class="error">BC laden mislukt: ${esc(e)}</div>`;
    }
  }

  refresh();
  window.setInterval(refresh, 5000);
}
"""


class BCPlugin(base_plugin.TBPlugin):
    plugin_name = "pokemonred_bc"

    def __init__(self, context):
        self._logdir = Path(context.logdir or "runs")

    def is_active(self):
        return True

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path="/index.js",
            tab_name="BC",
            disable_reload=False,
        )

    def get_plugin_apps(self):
        return {
            "/index.js": self._serve_js,
            "/data": self._serve_data,
        }

    @wrappers.Request.application
    def _serve_js(self, request):
        return http_util.Respond(request, _JS, "application/javascript")

    def _history_path(self):
        p = self._logdir
        if not p.is_absolute():
            p = Path.cwd() / p
        return p / "bc_tensorboard_history.jsonl"

    def _read_json(self, path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_history(self, max_items=200):
        path = self._history_path()
        if not path.exists():
            return []
        rows = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return rows[-max_items:]

    @wrappers.Request.application
    def _serve_data(self, request):
        snapshot = Path(
            os.environ.get(
                "POKEMON_BC_SNAPSHOT_PATH",
                f"/dev/shm/pokemonred_bc_{os.getuid()}.json",
            )
        )
        payload = {
            "current": self._read_json(snapshot),
            "history": self._read_history(),
        }
        return http_util.Respond(request, payload, "application/json")
