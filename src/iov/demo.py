"""Local dashboard and a portable HTML replay; no external web services needed."""
import csv
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .evaluation import episode, POLICIES
from .learner import Learner


class Demo:
    def __init__(self, run):
        self.run = Path(run)
        self.learner = Learner.load(self.run / "best.pt")
        self.untrained = Learner.load(self.run / "untrained.pt")
        self.config = self.learner.config.environment

    def rollout(self, policy="trained", seed=300000, workload=1.0):
        if policy not in POLICIES:
            raise ValueError("Unknown policy")
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("Seed must be between 0 and 2147483647")
        if not 0.25 <= workload <= 2.0:
            raise ValueError("Workload must be between 0.25 and 2.0")
        model = self.untrained if policy == "untrained" else self.learner
        return episode(self.config, policy, seed, model, frames=True, workload=workload)

    def payload(self, offline=False):
        result = {"config": asdict(self.config), "offline": offline,
                  "evaluation": json.loads((self.run / "evaluation.json").read_text()),
                  "validation": json.loads((self.run / "validation.json").read_text()),
                  "history": list(csv.DictReader((self.run / "history.csv").open())),
                  "manifest": json.loads((self.run / "training_manifest.json").read_text())}
        if offline:
            result["recordings"] = {p: self.rollout(p) for p in POLICIES}
        return result


def render(payload):
    template = (Path(__file__).parent / "web" / "index.html").read_text()
    # Protect the embedded script boundary even if metadata gains text fields.
    serialized = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__IOV_DATA__", serialized)


def export(run, output=None):
    output = Path(output) if output else Path(run) / "showcase.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(Demo(run).payload(offline=True)))
    print(f"Saved offline interactive showcase: {output}")
    return output


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    self.respond(200, render(demo.payload()).encode(), "text/html; charset=utf-8")
                elif request.path == "/api/health":
                    self.respond_json(200, {"status": "ok", "model_loaded": True, "agents": demo.config.agents})
                elif request.path == "/api/rollout":
                    params = parse_qs(request.query)
                    result = demo.rollout(params.get("policy", ["trained"])[0],
                                          int(params.get("seed", ["300000"])[0]),
                                          float(params.get("workload", ["1"])[0]))
                    self.respond_json(200, result)
                else:
                    self.respond_json(404, {"error": "Not found"})
            except (ValueError, TypeError) as exc:
                self.respond_json(400, {"error": str(exc)})

        def respond_json(self, status, value):
            self.respond(status, json.dumps(value).encode(), "application/json")

        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(run, host="127.0.0.1", port=8000):
    demo = Demo(run)
    server = ThreadingHTTPServer((host, port), handler_for(demo))
    print(f"Dashboard listening on {host}:{server.server_port}; Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

