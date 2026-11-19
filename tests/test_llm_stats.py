"""llm-stats against a fake Maya server: Live line, idle, missing /metrics."""
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "llm-stats"

ENGINE = {"model": "deepseek-v4.1-flash", "model_label": "DeepSeek V4.1 · UD-Q2_K_XL",
          "version": "ds4-0.1", "kv": "fp4"}
LIVE = {"state": "generating", "phase": "thinking", "generated": 6905, "max_tokens": 32000,
        "elapsed_s": 136.1, "tok_s": 70.8, "tok_s_mean": 52.2}


class MayaServer(HTTPServer):
    live = LIVE
    metrics_status = 200


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        routes = {
            "/health": {"status": "ok"},
            "/v1/status": {"service": "maya", "model": "deepseek-v4.1-flash"},
            "/v1/models": {"data": [{"id": "deepseek-v4.1-flash"}]},
            "/metrics": {"engine": ENGINE, "live": self.server.live},
            "/metrics/prometheus": "",
            "/slots": [{"id": 0, "n_ctx": 524288, "is_processing": True}],
        }
        bad = self.path not in routes or (self.path == "/metrics" and self.server.metrics_status != 200)
        self.send_response(404 if bad else 200)
        if not bad:
            body = routes[self.path]
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not bad:
            self.wfile.write(data)

    def log_message(self, *args):
        pass


def run_stats(port):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--once", "--port", str(port), "--host", "127.0.0.1"],
        capture_output=True, text=True, timeout=30, env=env)


class LlmStatsMayaTest(unittest.TestCase):
    def serve(self, **cfg):
        server = MayaServer(("127.0.0.1", 0), Handler)
        for k, v in cfg.items():
            setattr(server, k, v)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_generating_shows_live_line(self):
        out = run_stats(self.serve())
        self.assertEqual(out.returncode, 0, out.stderr)
        for want in ["Maya", "Live", "thinking", "6905", "32000", "70.8", "52.2", "136.1", "DeepSeek V4.1"]:
            self.assertIn(want, out.stdout)

    def test_idle_has_no_live_line(self):
        out = run_stats(self.serve(live={"state": "idle", "generated": None, "tok_s": None}))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Maya", out.stdout)
        self.assertNotIn("Live", out.stdout)

    def test_metrics_404_still_renders(self):
        out = run_stats(self.serve(metrics_status=404))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Maya", out.stdout)
        self.assertIn("Prefill", out.stdout)
        self.assertNotIn("Live", out.stdout)


BASALT_ENGINE = {"version": "0.4.1", "quant": "Q4_K_XL", "kv": "fp16"}
PROMETHEUS = (
    "# HELP llamacpp:prompt_tokens_total prompt tokens processed\n"
    'llamacpp:prompt_tokens_total{model="test"} 5000\n'
    'llamacpp:prompt_seconds_total{model="test"} 2.5\n'
    'llamacpp:tokens_predicted_total{model="test"} 900\n'
    'llamacpp:tokens_predicted_seconds_total{model="test"} 15.0\n'
)


class BasaltHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        routes = {
            "/health": {"service": "basalt", "status": "ok"},
            "/v1/models": {"data": [{"id": "basalt-test", "meta": {"n_ctx": 8192}}]},
            "/metrics": {"engine": BASALT_ENGINE,
                         "totals": {"prompt": 5000, "reused": 1200, "prompt_ms": 2500.0,
                                    "written": 900, "decode_ms": 15000.0},
                         "live": {"slots": {"active": 1}, "queued": 2}},
            "/slots": [{"id": 0, "n_ctx": 8192, "is_processing": True, "ctx_used": 512,
                        "tok_s": 42.5, "drafts_offered": 10, "drafts_accepted": 7}],
        }
        self.respond(routes)

    def respond(self, routes):
        bad = self.path not in routes
        self.send_response(404 if bad else 200)
        if not bad:
            body = routes[self.path]
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not bad:
            self.wfile.write(data)

    def log_message(self, *args):
        pass


class LlamaCppHandler(BasaltHandler):
    def do_GET(self):
        routes = {
            "/health": {"status": "ok"},
            "/v1/models": {"data": [{"id": "llama-test", "meta": {"n_ctx": 4096}}]},
            "/metrics": PROMETHEUS,
            "/slots": [],
        }
        self.respond(routes)


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class LlmStatsHeaderTest(unittest.TestCase):
    def test_basalt_header_has_engine_extras(self):
        server = serve(BasaltHandler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        out = run_stats(server.server_address[1])
        self.assertEqual(out.returncode, 0, out.stderr)
        head = out.stdout.splitlines()[0]
        for want in ["Basalt", "basalt 0.4.1", "Q4_K_XL", "kv fp16"]:
            self.assertIn(want, head)
        self.assertNotIn("Live", out.stdout)

    def test_llamacpp_has_no_live_line(self):
        server = serve(LlamaCppHandler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        out = run_stats(server.server_address[1])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("llama.cpp", out.stdout)
        self.assertNotIn("Live", out.stdout)


if __name__ == "__main__":
    unittest.main()
