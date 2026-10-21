"""Lokales Trainings-Dashboard (nur Python-Bordmittel, keine Extra-Pakete).

Start:
    .venv\\Scripts\\python.exe dashboard.py
    ->  http://localhost:7070

Was es tut:
  * Zeigt Live-Fortschritt aller Trainingslaeufe (liest out/<name>.log)
  * Zeigt die GPU-Auslastung (nvidia-smi)
  * Startet neue Laeufe (train_netcov.py) mit frei waehlbaren Werten
  * Editiert die Missions-Config (netzabdeckung_config.yaml) mit Backup
  * Listet laufende Prozesse und kann sie gezielt stoppen

Das Dashboard ist ein REINER Beobachter/Starter. Es fasst einen bereits
laufenden Prozess niemals von selbst an - Stoppen passiert nur per Klick.
Config-Aenderungen wirken erst beim NAECHSTEN Lauf (ein laufender Prozess hat
die YAML schon geladen).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CONFIG_DEFAULT = ROOT / "netzabdeckung_config.yaml"
PORT = 7070

# Python aus der venv (fuer gestartete Laeufe)
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYEXE = str(VENV_PY if VENV_PY.exists() else sys.executable)

UPDATE_RE = re.compile(
    r"Update\s+(\d+)\s*\|\s*Stufe\s+(\d+)\s*\|\s*served\s+([\d.]+)%\s*\|\s*"
    r"conn\s+([\d.]+)\s*\|\s*holes\s+([\d.]+)\s*\|\s*crash\s+([\d.]+)\s*\|\s*"
    r"batt\s+([\d.]+)%\s*\|\s*([\d,]+)\s*steps/s"
)


# --------------------------------------------------------------------------- #
# Datensammlung
# --------------------------------------------------------------------------- #
def gpu_info() -> dict:
    try:
        q = ("utilization.gpu,memory.used,memory.total,power.draw,"
             "power.limit,temperature.gpu,name")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        ).stdout.strip()
        if not out:
            return {"available": False}
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "available": True,
            "util": float(parts[0]), "mem_used": float(parts[1]),
            "mem_total": float(parts[2]), "power": float(parts[3]),
            "power_limit": float(parts[4]), "temp": float(parts[5]),
            "name": parts[6],
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}


def running_train_procs() -> list[dict]:
    """Laufende train_netcov-Prozesse via PowerShell/CIM (liefert Kommandozeile)."""
    ps = (
        "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'train_netcov' } | "
        "Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
    except Exception:  # noqa: BLE001
        return []

    procs = []
    for d in data:
        cmd = d.get("CommandLine", "") or ""
        procs.append({
            "pid": int(d["ProcessId"]),
            "cmd": cmd,
            "name": _arg(cmd, "--name"),
            "updates": _arg_int(cmd, "--updates", 3000),
            "envs": _arg_int(cmd, "--envs", 256),
            "stage": _arg_int(cmd, "--start-stage", 1),
            "config": _arg(cmd, "--config") or "netzabdeckung_config.yaml",
        })
    return procs


def _arg(cmd: str, flag: str) -> str | None:
    m = re.search(re.escape(flag) + r"\s+(\S+)", cmd)
    return m.group(1) if m else None


def _arg_int(cmd: str, flag: str, default: int) -> int:
    v = _arg(cmd, flag)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def parse_log(path: Path, tail_rows: int = 120) -> dict:
    """Liest eine out/<name>.log und zieht die Update-Zeilen raus."""
    if not path.exists():
        return {"rows": [], "done": False, "stages": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return {"rows": [], "done": False, "stages": []}
    rows = []
    for m in UPDATE_RE.finditer(text):
        rows.append({
            "u": int(m.group(1)), "stage": int(m.group(2)),
            "served": float(m.group(3)), "conn": float(m.group(4)),
            "holes": float(m.group(5)), "crash": float(m.group(6)),
            "batt": float(m.group(7)), "sps": int(m.group(8).replace(",", "")),
        })
    done = "Fertig." in text
    stages = [int(x) for x in re.findall(r"Aufstieg zu Stufe (\d+)", text)]
    return {"rows": rows[-tail_rows:], "done": done, "stages": stages}


def meta_path(name: str) -> Path:
    return OUT / f"{name}.meta.json"


def collect_runs() -> list[dict]:
    procs = running_train_procs()
    by_name = {p["name"]: p for p in procs if p["name"]}

    # Alle bekannten Laeufe: laufende + solche mit Log oder Meta-Datei
    names: set[str] = set(by_name)
    if OUT.exists():
        for f in OUT.glob("*.log"):
            if f.stem.endswith(".err"):
                continue
            names.add(f.stem)
        for f in OUT.glob("*.meta.json"):
            names.add(f.name[:-len(".meta.json")])

    runs = []
    for name in sorted(names):
        if name == "dashboard":          # eigenes Log des Dashboards nicht anzeigen
            continue
        log = parse_log(OUT / f"{name}.log")
        meta = {}
        mp = meta_path(name)
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        proc = by_name.get(name)
        # nur echte Laeufe: laufend ODER mit Meta-Datei ODER mit Trainingszeilen
        if proc is None and not meta and not log["rows"]:
            continue
        total = (proc or {}).get("updates") or meta.get("updates") or 0
        rows = log["rows"]
        cur = rows[-1]["u"] if rows else 0
        last = rows[-1] if rows else None
        runs.append({
            "name": name,
            "running": proc is not None,
            "pid": (proc or {}).get("pid"),
            "total": total,
            "cur": cur,
            "pct": round(100.0 * cur / total, 1) if total else 0.0,
            "done": log["done"],
            "envs": (proc or {}).get("envs") or meta.get("envs"),
            "stage": last["stage"] if last else (proc or {}).get("stage"),
            "config": (proc or {}).get("config") or meta.get("config"),
            "last": last,
            "rows": rows,
            "stages": log["stages"],
            "started": meta.get("started"),
        })
    # laufende zuerst, dann nach Name
    runs.sort(key=lambda r: (not r["running"], r["name"]))
    return runs


# --------------------------------------------------------------------------- #
# Aktionen
# --------------------------------------------------------------------------- #
def start_run(body: dict) -> dict:
    name = str(body.get("name") or f"run_{int(time.time())}").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or f"run_{int(time.time())}"
    updates = int(body.get("updates", 3000))
    envs = int(body.get("envs", 1024))
    stage = int(body.get("start_stage", 1))
    config = str(body.get("config") or "netzabdeckung_config.yaml")
    use_cpu = bool(body.get("cpu", False))

    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"{name}.log", "w", encoding="utf-8")  # noqa: SIM115

    args = [PYEXE, "-u", "train_netcov.py",
            "--updates", str(updates), "--envs", str(envs),
            "--start-stage", str(stage), "--name", name, "--config", config]
    if use_cpu:
        args.append("--cpu")

    # Detached starten, damit der Lauf das Dashboard ueberlebt
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        p = subprocess.Popen(
            args, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags, close_fds=True,
        )
    except Exception as e:  # noqa: BLE001
        logf.close()
        return {"error": str(e)}

    meta_path(name).write_text(json.dumps({
        "name": name, "updates": updates, "envs": envs, "stage": stage,
        "config": config, "cpu": use_cpu, "pid": p.pid,
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    return {"ok": True, "pid": p.pid, "name": name}


def stop_run(pid: int) -> dict:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                       capture_output=True, text=True, timeout=10)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def read_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {"error": f"nicht gefunden: {p}"}
    return {"text": p.read_text(encoding="utf-8"), "path": str(p),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}


def write_config(path: str, text: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        import yaml  # nur zum Validieren
        yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001
        return {"error": f"YAML ungueltig, nicht gespeichert: {e}"}
    backup = None
    if p.exists():
        backup = p.with_suffix(p.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(p, backup)
    p.write_text(text, encoding="utf-8")
    return {"ok": True, "backup": backup.name if backup else None}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # ruhig bleiben
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json({"gpu": gpu_info(), "runs": collect_runs(),
                        "now": datetime.now().isoformat(timespec="seconds")})
        elif path == "/api/config":
            cfg = urlparse(self.path).query
            cfgpath = "netzabdeckung_config.yaml"
            if cfg.startswith("path="):
                from urllib.parse import unquote
                cfgpath = unquote(cfg[len("path="):])
            self._json(read_config(cfgpath))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            body = {}
        if path == "/api/start":
            self._json(start_run(body))
        elif path == "/api/stop":
            self._json(stop_run(int(body.get("pid"))))
        elif path == "/api/config":
            self._json(write_config(body.get("path", "netzabdeckung_config.yaml"),
                                    body.get("text", "")))
        else:
            self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Schwarm-Training Dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2a3140;--txt:#e6edf3;
--mut:#8b949e;--acc:#3fb950;--acc2:#58a6ff;--warn:#d29922;--bad:#f85149;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--line);display:flex;
align-items:center;gap:14px}header h1{font-size:17px;margin:0;font-weight:650}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut)}
.dot.live{background:var(--acc);box-shadow:0 0 8px var(--acc)}
main{padding:22px;max-width:1150px;margin:0 auto;display:grid;gap:18px}
.grid{display:grid;gap:18px}.g2{grid-template-columns:1fr 1fr}
@media(max-width:800px){.g2{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);
margin:0 0 12px}
.row{display:flex;gap:10px;flex-wrap:wrap}
.metric{background:var(--panel2);border:1px solid var(--line);border-radius:9px;
padding:10px 12px;min-width:92px;flex:1}
.metric .v{font-size:20px;font-weight:650}.metric .l{color:var(--mut);font-size:11px}
.bar{height:10px;background:var(--panel2);border-radius:6px;overflow:hidden;margin:8px 0}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc2),var(--acc));width:0}
.run{border:1px solid var(--line);border-radius:10px;padding:13px;margin-bottom:12px;background:var(--panel2)}
.run.on{border-color:var(--acc)}
.run .top{display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
.tag.on{color:var(--acc);border-color:var(--acc)}.tag.done{color:var(--acc2);border-color:var(--acc2)}
canvas{width:100%;height:90px;display:block;margin-top:8px}
label{display:block;font-size:12px;color:var(--mut);margin:8px 0 3px}
input,select,textarea{width:100%;background:var(--bg);color:var(--txt);border:1px solid var(--line);
border-radius:7px;padding:8px 10px;font:inherit}
textarea{min-height:360px;font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:pre}
button{background:var(--acc);color:#06240f;border:0;border-radius:7px;padding:9px 16px;
font:inherit;font-weight:650;cursor:pointer}button:hover{filter:brightness(1.08)}
button.sec{background:var(--panel2);color:var(--txt);border:1px solid var(--line)}
button.danger{background:var(--bad);color:#fff}
.formgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
@media(max-width:600px){.formgrid{grid-template-columns:1fr 1fr}}
.msg{font-size:12px;margin-top:8px;min-height:16px}
.msg.ok{color:var(--acc)}.msg.err{color:var(--bad)}.msg.warn{color:var(--warn)}
small{color:var(--mut)}
.gaugewrap{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.gauge{flex:1;min-width:120px}
</style></head><body>
<header><span class="dot" id="pulse"></span><h1>Schwarm-Training Dashboard</h1>
<small id="clock"></small>
<small style="margin-left:auto">TensorBoard: <a style="color:var(--acc2)" href="http://localhost:6006" target="_blank">localhost:6006</a></small>
</header>
<main>

<div class="card" id="gpucard"><h2>GPU</h2><div class="gaugewrap" id="gpu">&mdash;</div></div>

<div class="card"><h2>Laeufe &amp; Fortschritt</h2><div id="runs">lade&hellip;</div></div>

<div class="grid g2">
  <div class="card"><h2>Neues Training starten</h2>
    <div class="formgrid">
      <div><label>Name</label><input id="s_name" placeholder="z.B. test_wind"></div>
      <div><label>Updates</label><input id="s_updates" type="number" value="3000"></div>
      <div><label>Envs (parallel)</label><input id="s_envs" type="number" value="1024"></div>
      <div><label>Start-Stufe</label>
        <select id="s_stage"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
      <div><label>Config</label><input id="s_config" value="netzabdeckung_config.yaml"></div>
      <div><label>Geraet</label><select id="s_cpu"><option value="0">GPU (cuda)</option><option value="1">CPU</option></select></div>
    </div>
    <div style="margin-top:12px"><button onclick="startRun()">Training starten</button></div>
    <div class="msg warn" id="s_msg">Hinweis: Ein zweiter GPU-Lauf teilt sich VRAM &amp; Rechenzeit mit dem aktuellen. Checkpoints (netcov_update*.pt) werden von allen Laeufen geteilt.</div>
  </div>

  <div class="card"><h2>Bedingungen aendern (Config)</h2>
    <label>Datei</label><input id="c_path" value="netzabdeckung_config.yaml">
    <div class="row" style="margin:8px 0"><button class="sec" onclick="loadCfg()">Laden</button>
      <button onclick="saveCfg()">Speichern (mit Backup)</button></div>
    <textarea id="c_text" spellcheck="false"></textarea>
    <div class="msg" id="c_msg"></div>
    <small>Aenderungen wirken erst beim naechsten Start. Der laufende Prozess hat die Werte bereits geladen.</small>
  </div>
</div>
</main>

<script>
const $=id=>document.getElementById(id);
function fmt(n,d=0){return n==null?"—":Number(n).toLocaleString("de-DE",{maximumFractionDigits:d,minimumFractionDigits:d})}

function spark(canvas,rows,key,color,max){
  const c=canvas.getContext("2d"),W=canvas.width=canvas.clientWidth*2,H=canvas.height=180;
  c.clearRect(0,0,W,H);if(!rows.length)return;
  const xs=rows.map(r=>r.u),mn=Math.min(...xs),mx=Math.max(...xs)||1;
  const top=max||Math.max(...rows.map(r=>r[key]),1);
  c.strokeStyle=color;c.lineWidth=3;c.beginPath();
  rows.forEach((r,i)=>{const x=(r.u-mn)/((mx-mn)||1)*(W-10)+5,y=H-8-(r[key]/top)*(H-20);
    i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
  c.fillStyle=color+"22";c.lineTo(W-5,H-8);c.lineTo(5,H-8);c.closePath();c.fill();
  c.fillStyle="#8b949e";c.font="22px system-ui";c.fillText(key+"  max "+fmt(top,0),8,24);
}

async function tick(){
  $("pulse").classList.add("live");$("clock").textContent=new Date().toLocaleTimeString("de-DE");
  let st;try{st=await (await fetch("/api/state")).json();}catch(e){$("pulse").classList.remove("live");return;}
  const g=st.gpu;
  if(g.available){
    const memp=100*g.mem_used/g.mem_total, powp=100*g.power/g.power_limit;
    $("gpu").innerHTML=`
      <div class="gauge"><div class="metric"><div class="v">${fmt(g.util)}%</div><div class="l">${g.name}</div></div></div>
      <div class="gauge"><div class="metric"><div class="v">${fmt(g.mem_used)} / ${fmt(g.mem_total)} MB</div><div class="l">VRAM ${fmt(memp)}%</div><div class="bar"><i style="width:${memp}%"></i></div></div></div>
      <div class="gauge"><div class="metric"><div class="v">${fmt(g.power)} / ${fmt(g.power_limit)} W</div><div class="l">Leistung ${fmt(powp)}%</div><div class="bar"><i style="width:${powp}%"></i></div></div></div>
      <div class="gauge"><div class="metric"><div class="v">${fmt(g.temp)} °C</div><div class="l">Temperatur</div></div></div>`;
  }else{$("gpu").innerHTML='<small>Keine NVIDIA-GPU gefunden (nvidia-smi). '+(g.error||"")+'</small>';}
  const host=$("runs");
  if(!st.runs.length){host.innerHTML="<small>Noch keine Laeufe. Starte unten einen.</small>";}
  else host.innerHTML=st.runs.map((r,i)=>{
    const L=r.last||{};
    const tag=r.running?'<span class="tag on">laeuft</span>':(r.done?'<span class="tag done">fertig</span>':'<span class="tag">gestoppt</span>');
    const E=eta(r);
    const etaCards=r.running?`
        <div class="metric" style="border-color:var(--acc2)"><div class="v">${E?E.remStr:'…'}</div><div class="l">Verbleibend</div></div>
        <div class="metric" style="border-color:var(--acc2)"><div class="v">${E?E.finishStr:'…'}</div><div class="l">Fertig gegen</div></div>`
      :(r.done?'<div class="metric"><div class="v">✓</div><div class="l">abgeschlossen</div></div>':'');
    return `<div class="run ${r.running?'on':''}">
      <div class="top"><b>${r.name}</b> ${tag}
        <span style="margin-left:auto">${r.running?`<button class="danger" onclick="stopRun(${r.pid},'${r.name}')">Stoppen (PID ${r.pid})</button>`:''}</span></div>
      <div class="bar"><i style="width:${r.pct}%"></i></div>
      <small>Update ${fmt(r.cur)} / ${fmt(r.total)} &nbsp;(${r.pct}%) &nbsp;|&nbsp; Stufe ${r.stage||'?'} &nbsp;|&nbsp; ${r.envs||'?'} envs${E?` &nbsp;|&nbsp; ${fmt(E.upm,0)} upd/min`:''}</small>
      <div class="row" style="margin-top:8px">
        ${etaCards}
        <div class="metric"><div class="v">${fmt(L.served,1)}%</div><div class="l">Abdeckung</div></div>
        <div class="metric"><div class="v">${fmt(L.batt,0)}%</div><div class="l">Akku Ø</div></div>
        <div class="metric"><div class="v">${fmt(L.holes,2)}</div><div class="l">Funkloecher</div></div>
        <div class="metric"><div class="v">${fmt(L.conn,2)}</div><div class="l">Uplink</div></div>
        <div class="metric"><div class="v">${fmt(L.crash,3)}</div><div class="l">Abstuerze/Schr.</div></div>
        <div class="metric"><div class="v">${fmt(L.sps)}</div><div class="l">steps/s</div></div>
      </div>
      <canvas id="cv_${i}"></canvas></div>`;
  }).join("");
  st.runs.forEach((r,i)=>{const cv=$("cv_"+i);if(cv)spark(cv,r.rows,"served","#3fb950",100);});
}

// Restzeit-Schaetzung ueber ein gleitendes Zeitfenster (robuster als 3s-Diff).
const hist={};
function eta(r){
  if(!r.running||!r.total||!r.cur)return null;
  const now=Date.now()/1000;
  const h=hist[r.name]=hist[r.name]||[];
  if(!h.length||h[h.length-1].cur!==r.cur)h.push({t:now,cur:r.cur}); // nur neue Werte
  while(h.length>2&&now-h[0].t>180)h.shift();                        // max 3 min Fenster
  const first=h[0];
  const span=now-first.t, prog=r.cur-first.cur;
  if(span<15||prog<=0)return null;         // noch zu wenig Daten fuer eine Schaetzung
  const ups=prog/span;                      // Updates pro Sekunde
  const remSec=(r.total-r.cur)/ups;
  const upm=ups*60;
  const remStr = remSec>5400 ? fmt(remSec/3600,1)+" h" : fmt(remSec/60,0)+" min";
  const fin=new Date(Date.now()+remSec*1000);
  const finishStr=fin.toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit"})+" Uhr";
  return {remSec,remStr,finishStr,upm};
}

async function startRun(){
  const b={name:$("s_name").value,updates:+$("s_updates").value,envs:+$("s_envs").value,
    start_stage:+$("s_stage").value,config:$("s_config").value,cpu:$("s_cpu").value==="1"};
  const m=$("s_msg");m.className="msg";m.textContent="starte…";
  const res=await (await fetch("/api/start",{method:"POST",body:JSON.stringify(b)})).json();
  if(res.ok){m.className="msg ok";m.textContent=`Gestartet: ${res.name} (PID ${res.pid})`;tick();}
  else{m.className="msg err";m.textContent="Fehler: "+(res.error||"unbekannt");}
}
async function stopRun(pid,name){
  if(!confirm(`Lauf "${name}" (PID ${pid}) wirklich stoppen?`))return;
  await fetch("/api/stop",{method:"POST",body:JSON.stringify({pid})});tick();
}
async function loadCfg(){
  const m=$("c_msg");m.className="msg";m.textContent="lade…";
  const res=await (await fetch("/api/config?path="+encodeURIComponent($("c_path").value))).json();
  if(res.error){m.className="msg err";m.textContent=res.error;return;}
  $("c_text").value=res.text;m.className="msg";m.textContent="Stand: "+res.mtime;
}
async function saveCfg(){
  const m=$("c_msg");m.className="msg";m.textContent="speichere…";
  const res=await (await fetch("/api/config",{method:"POST",
    body:JSON.stringify({path:$("c_path").value,text:$("c_text").value})})).json();
  if(res.ok){m.className="msg ok";m.textContent="Gespeichert"+(res.backup?` (Backup: ${res.backup})`:"")+" — wirkt beim naechsten Start.";}
  else{m.className="msg err";m.textContent=res.error;}
}
tick();setInterval(tick,3000);loadCfg();
</script></body></html>"""


def main():
    os.chdir(ROOT)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dashboard laeuft:  http://localhost:{PORT}")
    print("Beenden mit Strg+C.  (Startet/stoppt Laeufe getrennt vom Dashboard.)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard beendet.")
        srv.shutdown()


if __name__ == "__main__":
    main()
