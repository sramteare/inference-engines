"""llm-stats against stub Maya, Basalt and llama.cpp servers."""
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "bin" / "llm-stats"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

ENGINE = {"model": "deepseek-v4.1-flash", "model_label": "DeepSeek V4.1 · UD-Q2_K_XL",
          "version": "ds4-0.1", "kv": "fp4"}
LIVE = {"state": "generating", "phase": "thinking", "generated": 6905, "max_tokens": 32000,
        "elapsed_s": 136.1, "tok_s": 70.8, "tok_s_mean": 52.2}
PROMETHEUS_42 = 'llamacpp:tokens_predicted_total{model="m"} 42\n'


def live_line(stdout):
    """The single dashboard line that starts with "Live"."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith("Live")]
    if len(lines) != 1:
        raise AssertionError(f"expected exactly one Live line in:\n{stdout}")
    return lines[0]


def rate_line(stdout, prefix):
    """The single dashboard line that starts with the given prefix ("Prefill"/"Generate")."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"expected exactly one {prefix} line in:\n{stdout}")
    return lines[0]


class StubHandler(BaseHTTPRequestHandler):
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


class MayaServer(HTTPServer):
    live = LIVE
    metrics_status = 200
    metrics_body = None      # when set, /metrics answers this JSON value instead
    prometheus = ""          # body of /metrics/prometheus


class MayaHandler(StubHandler):
    def do_GET(self):
        if self.path == "/metrics" and self.server.metrics_body is not None:
            body = self.server.metrics_body      # a str is served verbatim: raw/invalid JSON text
            self.respond({"/metrics": body if isinstance(body, str) else json.dumps(body)})
            return
        routes = {
            "/health": {"status": "ok"},
            "/v1/status": {"service": "maya", "model": "deepseek-v4.1-flash"},
            "/v1/models": {"data": [{"id": "deepseek-v4.1-flash"}]},
            "/metrics": {"engine": ENGINE, "live": self.server.live},
            "/metrics/prometheus": self.server.prometheus,
            "/slots": [{"id": 0, "n_ctx": 524288, "is_processing": True}],
        }
        if self.path == "/metrics" and self.server.metrics_status != 200:
            routes.pop("/metrics")
        self.respond(routes)


def run_stats(port, script=SCRIPT):
    return subprocess.run(
        [sys.executable, str(script), "--once", "--port", str(port), "--host", "127.0.0.1"],
        capture_output=True, text=True, timeout=30, env=ENV)


class LlmStatsMayaTest(unittest.TestCase):
    def serve(self, **cfg):
        server = MayaServer(("127.0.0.1", 0), MayaHandler)
        for k, v in cfg.items():
            setattr(server, k, v)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_generating_shows_live_line(self):
        out = run_stats(self.serve())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Maya", out.stdout)
        self.assertIn("DeepSeek V4.1", out.stdout)
        line = live_line(out.stdout)
        for want in ["thinking", "generated 6905/32000 tok", "70.8 tok/s now", "mean 52.2", "elapsed 136.1 s"]:
            self.assertIn(want, line)

    def test_reading_shows_prompt_not_generated(self):
        live = {"state": "reading", "phase": "reading", "prompt_read": 1024, "prompt_total": 3072,
                "generated": None, "max_tokens": 32000, "elapsed_s": 4.2, "tok_s": None, "tok_s_mean": None}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        line = live_line(out.stdout)
        self.assertIn("prompt 1024/3072 tok", line)
        self.assertNotIn("generated", line)

    def test_generating_null_tok_s_omits_rate(self):
        live = {"state": "generating", "phase": "writing", "generated": 120, "max_tokens": 500,
                "elapsed_s": 2.0, "tok_s": None, "tok_s_mean": None}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        line = live_line(out.stdout)
        self.assertIn("generated 120/500 tok", line)
        self.assertNotIn("tok/s now", line)

    def test_reading_prompt_wins_over_generated(self):
        live = {"state": "reading", "prompt_read": 10, "prompt_total": 20, "generated": 5}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        line = live_line(out.stdout)
        self.assertIn("prompt 10/20 tok", line)
        self.assertNotIn("generated", line)

    def test_generating_generated_wins_over_prompt(self):
        live = {"state": "generating", "prompt_read": 10, "prompt_total": 20, "generated": 5}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        line = live_line(out.stdout)
        self.assertIn("generated 5", line)
        self.assertNotIn("prompt", line)

    def test_non_string_phase_omitted(self):
        live = {"state": "generating", "phase": 1, "generated": 2000, "max_tokens": 8000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        line = live_line(out.stdout)
        self.assertIn("generated 2000/8000 tok", line)
        self.assertNotIn("1", line)   # a leaked phase would render str(1) = "1"

    def test_empty_state_has_no_live_line(self):
        out = run_stats(self.serve(live={"state": "", "generated": 5}))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Maya", out.stdout)
        self.assertNotIn("Live", out.stdout)

    def test_empty_phase_has_no_blank_segment(self):
        out = run_stats(self.serve(live={"state": "generating", "phase": "", "generated": 5}))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(live_line(out.stdout), "Live       generated 5 tok")

    def test_huge_int_generated_never_renders(self):
        body = '{"live": {"state": "generating", "phase": "thinking", "generated": %s}}' % ("9" * 400)
        out = run_stats(self.serve(metrics_body=body))
        self.assertEqual(out.returncode, 0, f"raw=400-digit int\n{out.stdout}\n{out.stderr}")
        line = live_line(out.stdout)
        self.assertIn("thinking", line)
        self.assertNotIn("generated", line)

    def test_unlimited_max_tokens_has_no_slash(self):
        for mx in (0, -1):
            live = {"state": "generating", "generated": 10, "max_tokens": mx,
                    "tok_s": None, "tok_s_mean": None}
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, f"max_tokens={mx}\n{out.stderr}")
            line = live_line(out.stdout)
            self.assertIn("generated 10 tok", line)
            self.assertNotIn("10/", line)

    def test_malformed_metrics_bodies_still_render(self):
        for body in ["broken", {"engine": "x", "live": "y"},
                     {"live": {"state": "generating", "tok_s": "fast", "phase": 1}}]:
            out = run_stats(self.serve(metrics_body=body))
            self.assertEqual(out.returncode, 0, f"body={body!r}\n{out.stdout}\n{out.stderr}")
            self.assertIn("Maya", out.stdout)
            self.assertIn("Prefill", out.stdout)

    def test_non_finite_and_bool_generated_never_render(self):
        for raw in ["1e309", "NaN", "true"]:   # raw JSON text: parses to inf, nan, True
            body = '{"live": {"state": "generating", "phase": "thinking", "generated": %s}}' % raw
            out = run_stats(self.serve(metrics_body=body))
            self.assertEqual(out.returncode, 0, f"raw={raw}\n{out.stdout}\n{out.stderr}")
            line = live_line(out.stdout)
            for bad in ["inf", "nan", "generated"]:
                self.assertNotIn(bad, line, f"raw={raw}\n{line}")

    def test_invalid_json_metrics_body_renders_dashboard(self):
        out = run_stats(self.serve(metrics_body='{"live":'))   # invalid JSON syntax
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Maya", out.stdout)
        self.assertIn("Prefill", out.stdout)
        self.assertNotIn("Live", out.stdout)

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

    def test_metrics_404_prometheus_still_counted(self):
        out = run_stats(self.serve(metrics_status=404, prometheus=PROMETHEUS_42))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = [ln for ln in out.stdout.splitlines() if ln.startswith("Generate")]
        self.assertEqual(len(gen), 1, out.stdout)
        self.assertIn("42 tok", gen[0])

    def test_reading_prefill_shows_now_rate_and_in_flight(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_read": 12000,
                "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("1234.5", pre)
        self.assertIn("+12000/40000 in flight", pre)
        gen = rate_line(out.stdout, "Generate")
        self.assertTrue(gen.startswith("Generate   last "), gen)   # first column stays "last"
        self.assertNotIn("now", gen)

    def test_reading_in_flight_counts_only_fresh_tokens(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": 100000,
                "prompt_read": 100500, "prompt_total": 102000, "prompt_tokens": 102000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("+500/2000 in flight", pre)

    def test_reading_stale_prompt_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        for ptk in [None, 3000]:   # prompt_tokens not set yet, or prompt_total is the previous request's
            live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_read": 12000,
                    "prompt_total": 40000, "prompt_tokens": ptk}
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, f"prompt_tokens={ptk!r}\n{out.stderr}")
            line = rate_line(out.stdout, "Prefill")
            self.assertEqual(line, base)   # byte-identical to the idle run
            self.assertIn("last ", line)
            self.assertNotIn("in flight", line)

    def test_reading_bad_prefill_rate_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        for raw in ["null", "NaN", "true", "10" + "0" * 29]:   # raw text: NaN survives the JSON round trip
            body = ('{"live": {"state": "reading", "prefill_tok_s_mean": %s, "prompt_read": 12000, '
                    '"prompt_total": 40000, "prompt_tokens": 40000}}' % raw)
            out = run_stats(self.serve(metrics_body=body))
            self.assertEqual(out.returncode, 0, f"raw={raw}\n{out.stdout}\n{out.stderr}")
            line = rate_line(out.stdout, "Prefill")
            self.assertEqual(line, base)
            self.assertNotIn("in flight", line)

    def test_reading_missing_prompt_read_shows_now_without_in_flight(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("1234.5", pre)
        self.assertNotIn("in flight", pre)

    def test_generating_generate_shows_now_rate_and_in_flight(self):
        live = {"state": "generating", "tok_s": 31.2, "generated": 412}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("now ", gen)
        self.assertIn("31.2", gen)
        self.assertIn("+412 in flight", gen)
        pre = rate_line(out.stdout, "Prefill")
        self.assertTrue(pre.startswith("Prefill    last "), pre)
        self.assertNotIn("in flight", pre)

    def test_generating_generate_line_exact_format(self):
        live = {"state": "generating", "tok_s": 31.2, "generated": 412}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(rate_line(out.stdout, "Generate"),
                         "Generate   now      31.2 tok/s   avg        - tok/s   (- tok, +412 in flight)")

    def test_generating_prefill_shows_mean_as_now(self):
        # prefill_tok_s_mean stays set for the whole request: show it, not the previous request's "last"
        live = {"state": "generating", "tok_s": 31.2, "generated": 412, "prefill_tok_s_mean": 1500.0}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("1500.0", pre)
        self.assertNotIn("in flight", pre)

    def test_generating_missing_generated_shows_now_without_in_flight(self):
        live = {"state": "generating", "tok_s": 31.2}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("now ", gen)
        self.assertIn("31.2", gen)
        self.assertNotIn("in flight", gen)

    def test_idle_prefill_generate_use_last(self):
        out = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(out.returncode, 0, out.stderr)
        for prefix in ("Prefill", "Generate"):
            line = rate_line(out.stdout, prefix)
            self.assertNotIn("now", line)
            self.assertNotIn("in flight", line)

    def test_unloaded_prefill_generate_use_last(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        out = run_stats(self.serve(live={"state": "unloaded"}))
        self.assertEqual(out.returncode, 0, out.stderr)
        for prefix in ("Prefill", "Generate"):
            self.assertEqual(rate_line(out.stdout, prefix), rate_line(idle.stdout, prefix))

    def test_generating_bad_tok_s_falls_back_to_idle_generate_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Generate")
        for ts in [None, "fast", True]:
            out = run_stats(self.serve(live={"state": "generating", "tok_s": ts, "generated": 412}))
            self.assertEqual(out.returncode, 0, f"tok_s={ts!r}\n{out.stderr}")
            line = rate_line(out.stdout, "Generate")
            self.assertEqual(line, base)   # byte-identical to the idle run: port never appears here
            self.assertIn("last ", line)
            self.assertNotIn("in flight", line)


BASALT_ENGINE = {"version": "0.4.1", "quant": "Q4_K_XL", "kv": "fp16"}
PROMETHEUS = (
    "# HELP llamacpp:prompt_tokens_total prompt tokens processed\n"
    'llamacpp:prompt_tokens_total{model="test"} 5000\n'
    'llamacpp:prompt_seconds_total{model="test"} 2.5\n'
    'llamacpp:tokens_predicted_total{model="test"} 900\n'
    'llamacpp:tokens_predicted_seconds_total{model="test"} 15.0\n'
)


class BasaltHandler(StubHandler):
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


class LlamaCppHandler(StubHandler):
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
    # main@475d09b prints exactly these lines for these stubs (byte-identical to this branch)
    GOLDEN = {"basalt": "Basalt @ http://127.0.0.1:{port}   model: basalt-test   ctx: 8192"
                          "   basalt 0.4.1, Q4_K_XL, kv fp16",
              "llamacpp": "llama.cpp @ http://127.0.0.1:{port}   model: llama-test   ctx: 4096"}
    GOLDEN_RATES = {
        "basalt": {"Prefill": "Prefill    last        - tok/s   avg   1520.0 tok/s   (3800 tok, 1200 cached)",
                   "Generate": "Generate   last        - tok/s   avg     60.0 tok/s   (900 tok)"},
        "llamacpp": {"Prefill": "Prefill    last        - tok/s   avg   2000.0 tok/s   (5000 tok, - cached)",
                     "Generate": "Generate   last        - tok/s   avg     60.0 tok/s   (900 tok)"},
    }

    def assert_header_unchanged(self, handler, golden):
        server = serve(handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        out = run_stats(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        head = out.stdout.splitlines()[0]
        self.assertEqual(head, golden.format(port=port))   # byte-identical header to main
        return out

    def test_basalt_header_unchanged(self):
        out = self.assert_header_unchanged(BasaltHandler, self.GOLDEN["basalt"])
        self.assertNotIn("Live", out.stdout)

    def test_llamacpp_header_unchanged(self):
        out = self.assert_header_unchanged(LlamaCppHandler, self.GOLDEN["llamacpp"])
        self.assertIn("llama.cpp", out.stdout)
        self.assertNotIn("Live", out.stdout)

    def test_basalt_rate_lines_unchanged(self):
        out = self.assert_header_unchanged(BasaltHandler, self.GOLDEN["basalt"])
        for prefix, golden in self.GOLDEN_RATES["basalt"].items():
            self.assertEqual(rate_line(out.stdout, prefix), golden)

    def test_llamacpp_rate_lines_unchanged(self):
        out = self.assert_header_unchanged(LlamaCppHandler, self.GOLDEN["llamacpp"])
        for prefix, golden in self.GOLDEN_RATES["llamacpp"].items():
            self.assertEqual(rate_line(out.stdout, prefix), golden)


if __name__ == "__main__":
    unittest.main()
