"""Local browser controls for a native CUDA client; no external web assets."""

from __future__ import annotations

import copy
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

PAGE = b"""<!doctype html><meta charset="utf-8"><title>MoQSplat viewer</title>
<style>body{background:#121820;color:#edf4fb;font:16px system-ui;margin:24px}
img{max-width:100%;max-height:72vh;display:block;background:#000;touch-action:none}
button,input{font:inherit;margin:8px;padding:6px}pre{white-space:pre-wrap}</style>
<h1>MoQSplat</h1><p>Drag to look. Click the image, then use W/A/S/D to move;
Q/E moves down/up. Hold Shift to move faster.</p><img id="view" src="/stream" tabindex="0">
<button id="trace">Follow recorded trace</button>
<label>GPU budget (MiB) <input id="budget" type="number" min="64" value="2048"></label>
<button id="apply">Apply budget</button><pre id="status">Waiting for a base layer...</pre>
<script>
const view=document.querySelector('#view'), keys=new Set(); let drag=false,dx=0,dy=0,busy=false;
view.onpointerdown=e=>{drag=true;view.focus();view.setPointerCapture(e.pointerId)};
view.onpointerup=()=>drag=false;
view.onpointermove=e=>{if(drag){dx+=e.movementX;dy+=e.movementY}};
view.onkeydown=e=>{keys.add(e.code);e.preventDefault()};view.onkeyup=e=>keys.delete(e.code);
view.onblur=()=>keys.clear();
async function post(path,data){const r=await fetch(path,{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
if(!r.ok)throw Error(await r.text())}
document.querySelector('#trace').onclick=()=>post('/input',{trace:true});
document.querySelector('#apply').onclick=()=>post('/budget',{
mib:Number(document.querySelector('#budget').value)});
setInterval(async()=>{if(busy)return; const speed=keys.has('ShiftLeft') ? .3 : .06;
const input={right:(+keys.has('KeyD')-keys.has('KeyA'))*speed,
forward:(+keys.has('KeyW')-keys.has('KeyS'))*speed,up:(+keys.has('KeyE')-keys.has('KeyQ'))*speed,
yaw:-dx*.003,pitch:-dy*.003};dx=dy=0;
if(Object.values(input).some(x=>x)){busy=true;
try{await post('/input',input)}finally{busy=false}}},50);
setInterval(async()=>{try{const s=await(await fetch('/status')).json();
if(s.gpu_budget_bytes&&!document.querySelector('#budget').dataset.initialized){
document.querySelector('#budget').value=Math.round(s.gpu_budget_bytes/1048576);
document.querySelector('#budget').dataset.initialized='1'}
document.querySelector('#status').textContent=s.state||
`Frames: ${s.frames} | Received layers: ${s.received_layers}/${s.total_layers}\n`+
`Render: ${(s.frame_ms||0).toFixed(1)} ms | GPU: `+
`${Math.round(s.cuda_peak_allocated_bytes/1048576)} / `+
`${Math.round(s.gpu_budget_bytes/1048576)} MiB\n`+
`Reconnections: ${s.reconnects} | GPU loads: ${s.promotions}`
}catch(e){document.querySelector('#status').textContent=
'Viewer connection closed. Last image retained.'}},500);
</script>"""


class PoseController:
    """Use trace poses until local browser movement takes control."""

    def __init__(self, trace):
        if not trace:
            raise ValueError("camera trace is empty")
        self.trace = trace
        self.lock = threading.Lock()
        self.manual = False
        self.pose = copy.deepcopy(trace[0])
        self.input_events = 0
        self.last_input = None

    def sample(self, elapsed_ms):
        with self.lock:
            if not self.manual:
                pose = self.trace[0]
                for candidate in self.trace:
                    if candidate["timestamp_ms"] - self.trace[0]["timestamp_ms"] > elapsed_ms:
                        break
                    pose = candidate
                self.pose = copy.deepcopy(pose)
            result = copy.deepcopy(self.pose)
            result.update(_input_events=self.input_events, _last_input=self.last_input)
            return result

    def input(self, command):
        with self.lock:
            if command.get("trace") is True:
                self.manual = False
                self.input_events += 1
                self.last_input = time.monotonic()
                return
            values = [
                float(command.get(key, 0)) for key in ("right", "up", "forward", "yaw", "pitch")
            ]
            if not all(math.isfinite(value) and abs(value) <= 5 for value in values):
                raise ValueError("movement must be finite and bounded")
            right, up, forward, yaw, pitch = values
            view = np.asarray(self.pose["view_matrix"], dtype=np.float64)
            orientation = view[:3, :3].T
            position = np.asarray(self.pose["camera_position"], dtype=np.float64)
            position += orientation @ np.array([right, up, -forward])
            cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
            orientation = orientation @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            orientation = orientation @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
            view = np.eye(4)
            view[:3, :3] = orientation.T
            view[:3, 3] = -orientation.T @ position
            self.pose.update(camera_position=position.tolist(), view_matrix=view.tolist())
            self.manual = True
            self.input_events += 1
            self.last_input = time.monotonic()


class FrameHub:
    """Keep only the latest JPEG and status, regardless of browser speed."""

    def __init__(self, controller):
        self.controller = controller
        self.condition = threading.Condition()
        self.jpeg = b""
        self.sequence = 0
        self.status = {"state": "waiting for base layer"}
        self.pending_budget = None
        self.stopped = False

    def publish(self, image, status):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        with self.condition:
            self.jpeg = buffer.getvalue()
            self.sequence += 1
            self.status = dict(
                status, frames=self.sequence, manual_controls=self.controller.input_events
            )
            self.condition.notify_all()


def start_viewer(hub, port):
    """Bind loopback only; a remote browser may use an explicit SSH tunnel."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, kind="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self.reply(PAGE, "text/html; charset=utf-8")
            if self.path == "/status":
                with hub.condition:
                    payload = json.dumps(hub.status).encode()
                return self.reply(payload)
            if self.path != "/stream":
                return self.reply(b"not found", "text/plain", 404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with hub.condition:
                        hub.condition.wait_for(lambda: hub.stopped or hub.sequence != last)
                        if hub.stopped:
                            break
                        jpeg, last = hub.jpeg, hub.sequence
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if (
                    not 0 < length <= 16384
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("bounded JSON request required")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError("JSON object required")
                if self.path == "/input":
                    hub.controller.input(command)
                elif self.path == "/budget":
                    budget = int(command["mib"])
                    if not 64 <= budget <= 65536:
                        raise ValueError("GPU budget must be 64..65536 MiB")
                    with hub.condition:
                        hub.pending_budget = budget
                else:
                    return self.reply(b"not found", "text/plain", 404)
                self.reply(b'{"accepted":true}')
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(str(exc).encode(), "text/plain", 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    return server
