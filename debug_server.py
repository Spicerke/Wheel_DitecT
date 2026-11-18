"""
debug_server.py - tiny stdlib web server that streams autopilot state to a browser.

The device only serialises JSON (~10 Hz); all drawing happens in the browser, so it can be
viewed from a laptop without loading the small Linux box.

  GET  /         web/index.html
  GET  /map      map grid (sent once)
  GET  /events   Server-Sent Events: {"state": {...}, "samples": [...], "path": [...]?}
  POST /cmd      {"cmd": "goal", "x":, "y":} | {"cmd": "stop"} | {"cmd": "pose", ...}
"""

import json
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class DebugHub:
    """Thread-safe mailbox between the control loop and HTTP clients."""

    def __init__(self, map_payload):
        self.map_json = json.dumps(map_payload, separators=(",", ":")).encode()
        self.commands = queue.Queue()
        self._lock = threading.Lock()
        self._state = {}
        self._samples = deque(maxlen=1000)    # ~20 s at 50 Hz
        self._seq = 0
        self._path = []
        self._path_ver = 0

    def publish(self, state):
        with self._lock:
            self._state = state

    def add_sample(self, sample):
        with self._lock:
            self._seq += 1
            self._samples.append(sample)

    def set_path(self, path):
        with self._lock:
            self._path = [[round(x, 2), round(y, 2)] for x, y in path]
            self._path_ver += 1

    def snapshot(self, last_seq, last_ver):
        with self._lock:
            n = min(self._seq - last_seq, len(self._samples))
            msg = {"state": self._state,
                   "samples": list(self._samples)[len(self._samples) - n:] if n > 0 else []}
            if self._path_ver != last_ver:
                msg["path"] = self._path
            return msg, self._seq, self._path_ver


def _handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body=b""):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    self._send(200, "text/html; charset=utf-8", f.read())
            elif self.path == "/map":
                self._send(200, "application/json", hub.map_json)
            elif self.path == "/events":
                self._events()
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path != "/cmd":
                return self._send(404, "text/plain", b"not found")
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 4096:
                return self._send(400, "text/plain", b"bad length")
            try:
                cmd = json.loads(self.rfile.read(n))
                if not isinstance(cmd, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, "text/plain", b"bad json")
            hub.commands.put(cmd)
            self._send(204, "text/plain")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_seq, ver = max(0, hub._seq - 500), -1
            try:
                while True:
                    msg, last_seq, ver = hub.snapshot(last_seq, ver)
                    data = json.dumps(msg, separators=(",", ":")).encode()
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def serve(hub, host="127.0.0.1", port=8080):
    srv = ThreadingHTTPServer((host, port), _handler(hub))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
