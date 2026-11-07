"""
Local web dashboard (--web): the primary interface for CameraVision.

OpenCV's window caps what the HUD can be; a browser does not. The dashboard
streams the annotated video (MJPEG) alongside a live control panel, and it
adds four things the OpenCV window can't:

  - a confidence sparkline (the signing letter's score over time),
  - a gesture-history timeline (every fired gesture/sign, newest first),
  - a database browser (every gesture the program knows, filterable by
    language and type), and
  - a proposal-review panel: the autonomous researcher's staged candidates,
    approved or rejected with one click (this is the durable-learning loop -
    what you approve merges into the database for good).

Pure standard library (http.server), no dependencies. The video loop calls
publish() once per frame; the app wires the control endpoints to the same
key handling so browser clicks act exactly like keypresses.
"""

import http.server
import json
import socketserver
import threading
import time

import cv2

_STATE = {"jpeg": None, "info": {}, "keys": [], "viewers": 0}
_LOCK = threading.Lock()
# hooks the app installs so the dashboard can read live data / act
HOOKS = {"db": None, "pending": None, "resolve": None}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>CameraVision</title><style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0d1220;color:#dfe7f3;
      font:14px/1.5 'Segoe UI',system-ui,sans-serif;display:flex;height:100vh}
 #left{flex:1;display:flex;flex-direction:column;background:#000;min-width:0}
 #left img{width:100%;height:100%;object-fit:contain}
 #panel{width:340px;overflow-y:auto;padding:16px;background:#141c2e}
 h1{font-size:19px;margin:0 0 2px;color:#6fb3ff} h1 span{color:#dfe7f3}
 .sub{color:#7b8aa6;font-size:12px;margin-bottom:12px}
 .word{background:#0b1120;border:1px solid #2a3550;border-radius:8px;padding:10px;
       font-size:24px;letter-spacing:3px;min-height:32px;color:#ffd479;
       margin-bottom:6px;word-break:break-word}
 canvas{width:100%;height:44px;background:#0b1120;border-radius:6px;
        margin-bottom:10px}
 .stat{color:#9fb0c8;font-size:12.5px;margin:2px 0}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:12px 0}
 button{background:#243250;border:0;border-radius:7px;color:#dfe7f3;
        padding:9px 6px;font-size:12.5px;cursor:pointer}
 button:hover{background:#31456e}
 .wide{grid-column:1/3}
 h2{font-size:13px;color:#6fb3ff;text-transform:uppercase;letter-spacing:.5px;
    margin:16px 0 6px;border-bottom:1px solid #24304a;padding-bottom:4px}
 .tl{max-height:150px;overflow-y:auto;font-size:12.5px}
 .tl div{padding:2px 0;border-bottom:1px solid #1a2336;color:#b9c6dc}
 .tl .t{color:#6b7a96;margin-right:6px}
 select{width:100%;background:#0b1120;color:#dfe7f3;border:1px solid #2a3550;
        border-radius:6px;padding:6px;margin-bottom:6px}
 .db{max-height:200px;overflow-y:auto;font-size:12px}
 .db div{display:flex;justify-content:space-between;padding:3px 0;
         border-bottom:1px solid #1a2336}
 .db .lang{color:#7b8aa6;font-size:11px}
 .pend{background:#1a2032;border:1px solid #33406a;border-radius:7px;
       padding:8px;margin-bottom:7px;font-size:12.5px}
 .pend .nm{color:#ffd479;font-weight:600}
 .pend .rz{color:#9aa7c0;font-size:11.5px;margin:3px 0}
 .pend button{padding:5px 10px;font-size:12px;margin-right:6px}
 .ok{background:#2c6e49}.no{background:#7a3245}
 .empty{color:#6b7a96;font-style:italic;font-size:12px}
</style></head><body>
<div id="left"><img src="/stream" alt="live stream"></div>
<div id="panel">
 <h1>Camera<span>Vision</span></h1>
 <div class="sub">live dashboard &mdash; the primary interface</div>
 <div class="word" id="word">-</div>
 <canvas id="spark" width="308" height="44"></canvas>
 <div class="stat" id="fps"></div>
 <div class="stat" id="gestures"></div>
 <div class="stat" id="modes"></div>
 <div class="row">
  <button onclick="k('d')">Objects (D)</button>
  <button onclick="k('g')">Gestures (G)</button>
  <button onclick="k('x')">Face (X)</button>
  <button onclick="k('l')">Spell mode (L)</button>
  <button class="ok" onclick="k('e')">English SL (E)</button>
  <button class="ok" onclick="k('t')">Turkish SL (T)</button>
  <button onclick="k('v')">Voice (V)</button>
  <button onclick="k('c')">Calibrate (C)</button>
  <button onclick="k('r')">Reload DB (R)</button>
  <button onclick="k('s')">Snapshot (S)</button>
  <button onclick="k('f')">Filter (F)</button>
  <button onclick="k('bksp')">Clear word</button>
 </div>
 <h2>Gesture history</h2>
 <div class="tl" id="timeline"><span class="empty">nothing yet</span></div>
 <h2>Knowledge base</h2>
 <select id="dbfilter" onchange="loadDB()">
  <option value="">all languages &amp; types</option>
  <option value="lang:ASL">ASL</option>
  <option value="lang:TID">Turkish (TİD)</option>
  <option value="type:word_sign">word signs</option>
  <option value="type:single_hand">hand poses</option>
  <option value="type:expression">expressions</option>
 </select>
 <div class="db" id="db"></div>
 <h2>Research proposals</h2>
 <div id="pending"><span class="empty">none staged</span></div>
</div>
<script>
const hist=[];
function k(c){fetch('/key?k='+encodeURIComponent(c)).then(()=>setTimeout(refresh,200))}
function spark(v){
 hist.push(v);if(hist.length>80)hist.shift();
 const c=document.getElementById('spark'),x=c.getContext('2d');
 x.clearRect(0,0,c.width,c.height);
 x.strokeStyle='#3aa0ff';x.lineWidth=2;x.beginPath();
 hist.forEach((h,i)=>{const px=i/79*c.width,py=c.height-h*c.height*0.9-2;
  i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
 x.fillStyle='#6b7a96';x.font='10px sans-serif';x.fillText('letter confidence',4,11);
}
async function refresh(){
 try{const s=await(await fetch('/info')).json();
  document.getElementById('word').textContent=s.word||'-';
  document.getElementById('fps').textContent='FPS '+(s.fps||0)+'  |  conf '+(s.conf||'');
  document.getElementById('gestures').textContent='Active: '+((s.gestures||[]).join(', ')||'-');
  document.getElementById('modes').textContent='Modes: '+(s.modes||'');
  spark(s.letter_score||0);
  const tl=document.getElementById('timeline');
  if(s.history&&s.history.length){tl.innerHTML=s.history.slice().reverse().map(
    e=>'<div><span class="t">'+e[0]+'</span>'+e[1]+'</div>').join('')}
 }catch(e){}
}
async function loadDB(){
 const f=document.getElementById('dbfilter').value;
 try{const rows=await(await fetch('/db?f='+encodeURIComponent(f))).json();
  document.getElementById('db').innerHTML=rows.length?rows.map(r=>
   '<div><span>'+r.name+'</span><span class="lang">'+(r.language||r.type)+'</span></div>'
  ).join(''):'<span class="empty">none</span>';
 }catch(e){}
}
async function loadPending(){
 try{const ps=await(await fetch('/pending')).json();
  const el=document.getElementById('pending');
  el.innerHTML=ps.length?ps.map(p=>
   '<div class="pend"><div class="nm">'+p.name+'</div>'+
   '<div class="rz">'+(p.reason||'')+'</div>'+
   '<button class="ok" onclick="resolve(\\''+p.name.replace(/'/g,"")+'\\',\\'approve\\')">Approve</button>'+
   '<button class="no" onclick="resolve(\\''+p.name.replace(/'/g,"")+'\\',\\'reject\\')">Reject</button></div>'
  ).join(''):'<span class="empty">none staged</span>';
 }catch(e){}
}
async function resolve(name,action){
 await fetch('/review?name='+encodeURIComponent(name)+'&action='+action);
 loadPending();loadDB();
}
refresh();loadDB();loadPending();
setInterval(refresh,500);setInterval(loadPending,4000);
</script></body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p == "/" or p.startswith("/index"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream":
            self._stream()
        elif p == "/info":
            with _LOCK:
                self._json(_STATE["info"])
        elif p.startswith("/db"):
            self._json(self._db(_qs(p).get("f", "")))
        elif p == "/pending":
            self._json(HOOKS["pending"]() if HOOKS["pending"] else [])
        elif p.startswith("/review"):
            q = _qs(p)
            res = ({"ok": False, "msg": "no resolver"} if not HOOKS["resolve"]
                   else HOOKS["resolve"](q.get("name", ""), q.get("action", "")))
            self._json(res)
        elif p.startswith("/key"):
            key = _qs(p).get("k", "")
            key = {"bksp": "\x08"}.get(key, key)
            if key:
                with _LOCK:
                    _STATE["keys"].append(key[0])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def _db(self, f):
        rows = HOOKS["db"]() if HOOKS["db"] else []
        if f.startswith("lang:"):
            want = f.split(":", 1)[1]
            rows = [r for r in rows if r.get("language") == want]
        elif f.startswith("type:"):
            want = f.split(":", 1)[1]
            rows = [r for r in rows if r.get("type") == want]
        return rows

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        with _LOCK:
            _STATE["viewers"] += 1
        try:
            while True:
                with _LOCK:
                    jpeg = _STATE["jpeg"]
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (ConnectionError, BrokenPipeError, OSError):
            return
        finally:
            with _LOCK:
                _STATE["viewers"] = max(0, _STATE["viewers"] - 1)


def _qs(path):
    if "?" not in path:
        return {}
    import urllib.parse
    return {k: v[0] for k, v in
            urllib.parse.parse_qs(path.split("?", 1)[1]).items()}


def start(port=8765, db=None, pending=None, resolve=None):
    """Serve the dashboard. db/pending/resolve are callables the app wires
    so the dashboard can read the database, list staged proposals, and
    approve/reject them."""
    HOOKS["db"], HOOKS["pending"], HOOKS["resolve"] = db, pending, resolve
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), _Handler)
    srv.daemon_threads = True
    srv.allow_reuse_address = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def publish(frame_bgr, info):
    """Push the latest annotated frame + status. Skips the JPEG encode
    entirely while nobody is streaming."""
    with _LOCK:
        _STATE["info"] = info
        watching = _STATE["viewers"] > 0
    if not watching:
        return
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if ok:
        with _LOCK:
            _STATE["jpeg"] = buf.tobytes()


def pop_keys():
    with _LOCK:
        keys, _STATE["keys"] = _STATE["keys"], []
    return keys
