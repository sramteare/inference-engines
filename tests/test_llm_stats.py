"""llm-stats against stub Maya, Basalt and llama.cpp servers."""
import contextlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
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

# Maya /metrics while two thinking requests are resumed: reused > prompt (the reused count holds the
# earlier output too) and engine_generated/decode_ms cover only each request's last segment
MAYA_RESUMED = json.loads((ROOT / "tests" / "data" / "maya-metrics-resumed.json").read_text())


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


def avg_tok_s(line):
    """The avg tok/s figure on a rate line, as a float; raises if it is a dash.
    Tolerates the Maya "avg(N)" label, so it reads the number just before the first "tok/s"."""
    seg = line.split("avg", 1)[1].split("tok/s")[0]
    return float(seg.split()[-1])


def avg_req_count(line):
    """The "avg(N)" count Maya labels its avg with, as an int; fails if there is no such label."""
    m = re.search(r"avg\((\d+)\)", line)
    assert m, f"no avg(N) label in {line}"
    return int(m.group(1))


class StubHandler(BaseHTTPRequestHandler):
    def respond(self, routes):
        self.server.hits = getattr(self.server, "hits", 0) + 1
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
    hits = 0               # requests answered: proves that importing llm-stats fetches nothing
    live = LIVE
    metrics_status = 200
    metrics_body = None      # when set, /metrics answers this JSON value instead
    prometheus = ""          # body of /metrics/prometheus
    requests = []            # /metrics "requests": finished requests, newest first


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
            "/metrics": {"engine": ENGINE, "live": self.server.live, "requests": self.server.requests},
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


def load_stats_module(port, script=SCRIPT):
    """Import llm-stats in-process against a live stub, --engine maya. Importing runs nothing: no argument
    parsing (sys.argv is never touched) and no polling loop; configure() applies the options the same way
    main() does when run as a script, then render() draws the first frame; later frames are driven by
    mutating the stub and calling mod.render()."""
    import importlib.util
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("llm_stats_under_test", str(script))
    spec = importlib.util.spec_from_loader("llm_stats_under_test", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)   # module level only defines: nothing is parsed, fetched or printed
    mod.configure(mod.p.parse_args(["--engine", "maya", "--host", "127.0.0.1", "--port", str(port)]))
    return mod, mod.render()


def rate_counts(stdout):
    """The (prefill_prompt, prefill_cached, generate_output) token counts currently on the dashboard.
    None where the dashboard shows a dash (Maya's prompt total has no Prometheus fallback).
    The counts sit in the LAST parentheses on the line (the "avg(N)" label has its own), as "P tok, C cached",
    "P, C cached" or "O tok". Maya may compact them to k/M to fit 80 columns: these tests use small counts and
    a compacted one fails here rather than silently comparing rounded values."""
    def counts(line):
        inner = line[line.rindex("(") + 1:line.rindex(")")]
        out = []
        for word in re.split(r"[ ,]", inner):
            if word in ("", "tok", "cached"):
                continue
            assert re.fullmatch(r"-|\d+", word), f"not an exact count: {word!r} in {line}"
            out.append(None if word == "-" else int(word))
        return out
    pre, gen = counts(rate_line(stdout, "Prefill")), counts(rate_line(stdout, "Generate"))
    return (pre[0], pre[1], gen[0])


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
        for want in ["thinking", "generated 6905/32000 tok", "70.8 tok/s", "(mean  52.2)", "elapsed  136.1 s"]:
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
        self.assertNotIn("tok/s", line)

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

    def test_empty_phase_keeps_columns(self):
        out = run_stats(self.serve(live={"state": "generating", "phase": "", "generated": 5}))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(live_line(out.stdout), "Live       " + " " * (10 + 2 + 12 + 2 + 12 + 2 + 16 + 2) + "generated 5 tok")

    def test_live_columns_do_not_move(self):
        lives = [{"state": "generating", "phase": "thinking", "generated": 5, "max_tokens": 32000,
                  "elapsed_s": 1.0, "tok_s": 9.5, "tok_s_mean": 8.0},
                 {"state": "generating", "phase": "writing", "generated": 123456, "max_tokens": 0,
                  "elapsed_s": 9999.9, "tok_s": 1234.5, "tok_s_mean": 999.9},
                 {"state": "generating", "phase": "thinking", "generated": 7, "elapsed_s": 3.0,
                  "tok_s": None, "tok_s_mean": None}]
        cols = set()
        for live in lives:
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, out.stderr)
            line = live_line(out.stdout)
            cols.add((line.index("elapsed"), line.index("generated")))
        self.assertEqual(len(cols), 1, cols)

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

    def test_reading_prefill_shows_now_rate_and_in_flight_suffix(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_read": 12000,
                "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("1234.5", pre)
        self.assertIn("+12k/40k in flight", pre)   # compact: the exact pair would not fit the line
        gen = rate_line(out.stdout, "Generate")
        self.assertTrue(gen.startswith("Generate   last "), gen)   # first column stays "last"
        self.assertNotIn("now", gen)

    def test_reading_in_flight_suffix_counts_only_fresh_tokens(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": 100000,
                "prompt_read": 100500, "prompt_total": 102000, "prompt_tokens": 102000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("+500/2.0k in flight", pre)   # 500 fresh of 2000 (102000 - 100000 reused), not of 102000

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

    def test_reading_missing_prompt_read_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertEqual(pre, base)
        self.assertNotIn("now", pre)
        self.assertNotIn("in flight", pre)

    def test_generating_generate_shows_now_rate_and_in_flight_suffix(self):
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
                         "Generate   now      31.2 tok/s   avg            - tok/s  (- tok, +412 in flight)")

    def test_generating_prefill_line_stays_idle_form(self):
        # no generating-phase Prefill "now": the line is the plain last/avg idle form
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        live = {"state": "generating", "tok_s": 31.2, "generated": 412, "prefill_tok_s_mean": 1500.0,
                "elapsed_s": 10.0}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertEqual(pre, base)
        self.assertIn("last ", pre)
        self.assertNotIn("now", pre)
        self.assertNotIn("in flight", pre)

    def test_generating_missing_generated_shows_now_without_in_flight_suffix(self):
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

    def test_reading_finished_prompt_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        for read in [40000, 39999]:   # a finished read ends at total, or total - 1 with the last token held back
            live = {"state": "reading", "prefill_tok_s_mean": 1300.0, "prompt_read": read,
                    "prompt_total": 40000, "prompt_tokens": 40000}
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, f"prompt_read={read}\n{out.stderr}")
            pre = rate_line(out.stdout, "Prefill")
            self.assertEqual(pre, base)   # stale: the read can no longer grow
            self.assertNotIn("now", pre)
            self.assertNotIn("in flight", pre)

    def test_reading_non_positive_prefill_rate_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        for pv in [0, -5]:
            live = {"state": "reading", "prefill_tok_s_mean": pv, "prompt_read": 100,
                    "prompt_total": 200, "prompt_tokens": 200}
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, f"prefill={pv!r}\n{out.stderr}")
            pre = rate_line(out.stdout, "Prefill")
            self.assertEqual(pre, base)
            self.assertNotIn("now", pre)
            self.assertNotIn("in flight", pre)

    def test_reading_prompt_read_past_total_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_read": 50000,
                "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertEqual(pre, base)
        self.assertNotIn("now", pre)
        self.assertNotIn("in flight", pre)

    def test_reading_negative_reused_does_not_subtract(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": -1,
                "prompt_read": 100, "prompt_total": 200, "prompt_tokens": 200}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("+100/200 in flight", pre)

    def test_reading_negative_prompt_read_falls_back_to_idle_prefill_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Prefill")
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_read": -1,
                "prompt_total": 40000, "prompt_tokens": 40000}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertEqual(pre, base)   # negative prompt_read: no live prefill at all
        self.assertNotIn("now", pre)
        self.assertNotIn("in flight", pre)

    def test_reading_reused_covering_the_prompt_shows_now_without_in_flight_suffix(self):
        for ru, read, total in [(40000, 20000, 40000),    # reused == total
                                (200000, 100500, 102000)]:   # reused past total
            live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": ru,
                    "prompt_read": read, "prompt_total": total, "prompt_tokens": total}
            out = run_stats(self.serve(live=live))
            self.assertEqual(out.returncode, 0, f"prompt_reused={ru}\n{out.stderr}")
            pre = rate_line(out.stdout, "Prefill")
            self.assertIn("now ", pre)
            self.assertIn("1234.5", pre)
            self.assertNotIn("in flight", pre)   # nothing fresh left to count

    def test_reading_reused_above_read_clamps_fresh_read_to_zero(self):
        live = {"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": 150,
                "prompt_read": 100, "prompt_total": 200, "prompt_tokens": 200}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("now ", pre)
        self.assertIn("+0/50 in flight", pre)

    def test_generating_negative_tok_s_falls_back_to_idle_generate_line(self):
        idle = run_stats(self.serve(live={"state": "idle"}))
        self.assertEqual(idle.returncode, 0, idle.stderr)
        base = rate_line(idle.stdout, "Generate")
        live = {"state": "generating", "tok_s": -1, "generated": 412}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertEqual(gen, base)
        self.assertIn("last ", gen)
        self.assertNotIn("in flight", gen)

    def test_generating_zero_tok_s_shows_now(self):
        live = {"state": "generating", "tok_s": 0.0, "generated": 3}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("now ", gen)
        self.assertIn("0.0 tok/s", gen)
        self.assertIn("+3 in flight", gen)

    def test_generating_negative_zero_tok_s_shows_zero(self):
        body = '{"live": {"state": "generating", "tok_s": -0.0, "generated": 3}}'   # raw text: -0.0 survives parsing
        out = run_stats(self.serve(metrics_body=body))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("now ", gen)
        self.assertIn("0.0 tok/s", gen)
        self.assertNotIn("-0.0", gen)

    def test_generating_negative_generated_shows_now_without_in_flight_suffix(self):
        live = {"state": "generating", "tok_s": 31.2, "generated": -1}
        out = run_stats(self.serve(live=live))
        self.assertEqual(out.returncode, 0, out.stderr)
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("now ", gen)
        self.assertIn("31.2", gen)
        self.assertNotIn("in flight", gen)

    def test_resumed_maya_totals_show_no_negative_and_pairs_drive_generate(self):
        # the saved bug moment: reused > prompt in totals (a resumed request's reused holds the earlier output
        # too), and engine_generated/decode_ms cover only each request's last segment. Nothing may go negative:
        # the Prefill count is Maya's growing prompt total 37939 (never prompt minus reused, which is negative
        # here), Prefill last/avg are "-" (both requests are visibly multi-call and reused > prompt),
        # Generate last is the newest request's own decode_tok_s, and Generate avg is the honest pair ratio
        # 1405 / 23.5 s = 59.8 labelled as its 2 requests; the Generate count is the true output total 66983.
        out = run_stats(self.serve(metrics_body=MAYA_RESUMED))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIsNone(re.search(r"-\d", pre), pre)   # never a negative count
        self.assertEqual(pre, "Prefill    last        - tok/s   avg            - tok/s"
                              "  (37939, 103515 cached)")   # 79 columns: no room for the word "tok"
        gen = rate_line(out.stdout, "Generate")
        self.assertIsNone(re.search(r"-\d", gen), gen)
        self.assertIn("last     54.5 tok/s", gen)         # the newest request's own decode rate
        self.assertAlmostEqual(avg_tok_s(gen), 59.8, delta=0.1)
        self.assertEqual(avg_req_count(gen), 2)           # the label says which requests the avg covers
        self.assertIn("(66983 tok)", gen)                 # the running output total, not the pair sum

    def test_normal_maya_totals_match_hand_computation(self):
        # reused < prompt for every request: the Prefill count is Maya's prompt total 5000 with 1000 cached;
        # Prefill avg is the fresh-over-ms ratio (5000-1000) / 2.5 s = 1600 over 1 request (no prefill_tok_s so
        # last is "-"); Generate last is the newest request's decode_tok_s 60, avg 900 / 15 s = 60 over 1
        # request, and the Generate count is output_tokens.
        body = {"engine": ENGINE, "live": {"state": "idle"},
                "requests": [{"prompt_tokens": 5000, "reused": 1000, "output_tokens": 900,
                              "engine_generated": 900, "prompt_ms": 2500.0, "decode_ms": 15000.0,
                              "decode_tok_s": 60.0}],
                "totals": {"requests": 1, "prompt_tokens": 5000, "reused": 1000, "output_tokens": 900,
                           "prompt_ms": 2500.0, "decode_ms": 15000.0}}
        prom = ('llamacpp:prompt_tokens_total{model="m"} 4000\n'
                'llamacpp:prompt_tokens_cached_total{model="m"} 1000\n'
                'llamacpp:prompt_seconds_total{model="m"} 2.5\n'
                'llamacpp:tokens_predicted_total{model="m"} 900\n'
                'llamacpp:tokens_predicted_seconds_total{model="m"} 15.0\n')
        out = run_stats(self.serve(metrics_body=body, prometheus=prom))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(rate_line(out.stdout, "Prefill"),
                         "Prefill    last        - tok/s   avg(1)    1600.0 tok/s  (5000 tok, 1000 cached)")
        self.assertEqual(rate_line(out.stdout, "Generate"),
                         "Generate   last     60.0 tok/s   avg(1)      60.0 tok/s  (900 tok)")

    def test_maya_generate_avg_dashes_when_no_pairs(self):
        # a resumed request with no engine_generated: no consistent pair, so the Generate avg is "-" (no bogus
        # totals avg), while last stays the request's own 54.5; the row is "unknown" (output 33512 with no
        # generated) with reused > prompt, so the Prefill last/avg dash, and the Prefill count is the growing
        # prompt total 17651 (the mirror's -32795 = prompt minus reused never reaches the dashboard)
        body = {"engine": ENGINE, "live": {"state": "idle"},
                "requests": [{"prompt_tokens": 17651, "reused": 50446, "output_tokens": 33512,
                              "decode_ms": 13137.2, "decode_tok_s": 54.5}],
                "totals": {"requests": 1, "prompt_tokens": 17651, "reused": 50446, "output_tokens": 33512,
                           "prompt_ms": 181.0, "decode_ms": 13137.2}}
        prom = ('llamacpp:prompt_tokens_total{model="m"} -32795\n'
                'llamacpp:prompt_tokens_cached_total{model="m"} 50446\n'
                'llamacpp:prompt_seconds_total{model="m"} 0.181\n'
                'llamacpp:tokens_predicted_total{model="m"} 33512\n'
                'llamacpp:tokens_predicted_seconds_total{model="m"} 13.1372\n')
        out = run_stats(self.serve(metrics_body=body, prometheus=prom))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIsNone(re.search(r"-\d", pre), pre)
        self.assertEqual(pre, "Prefill    last        - tok/s   avg            - tok/s"
                              "  (17651, 50446 cached)")
        self.assertEqual(rate_line(out.stdout, "Generate"),
                         "Generate   last     54.5 tok/s   avg            - tok/s  (33512 tok)")

    def _maya_server(self):
        server = MayaServer(("127.0.0.1", 0), MayaHandler)
        server.live = {"state": "idle"}
        server.prometheus = ""    # counts come from the JSON totals, not the Prometheus mirror
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    @staticmethod
    def _req(k):
        prompt, reused, out, gen = 2000 + 100 * k, 500 + 50 * k, 150 + 10 * k, 150 + 10 * k
        pms, dms = 1000.0 + 10 * k, 3000.0 + 20 * k
        return {"prompt_tokens": prompt, "reused": reused, "output_tokens": out,
                "engine_generated": gen, "prompt_ms": pms, "decode_ms": dms,
                "decode_tok_s": round(gen / (dms / 1000), 1),
                "prefill_tok_s": round((prompt - reused) / (pms / 1000), 1)}

    @staticmethod
    def _totals(reqs):
        return {"requests": len(reqs),
                "prompt_tokens": sum(r["prompt_tokens"] for r in reqs),
                "reused": sum(r["reused"] for r in reqs),
                "output_tokens": sum(r["output_tokens"] for r in reqs),
                "prompt_ms": sum(r["prompt_ms"] for r in reqs),
                "decode_ms": sum(r["decode_ms"] for r in reqs)}

    def _body(self, window, seen):
        return {"engine": ENGINE, "live": {"state": "idle"}, "requests": window,
                "totals": self._totals(seen)}   # totals accumulate over every request ever seen

    def test_maya_multi_frame_rolling_window_never_negative_or_receding(self):
        # Maya keeps only the newest 12 requests; across frames a 13th request arrives and pushes the oldest
        # out of that window. Rates never use frame-to-frame counter differences, so a rolling-out request must
        # not make a rate negative, and the running token counts must not decrease.
        server = self._maya_server()
        base = [self._req(k) for k in range(12)]        # 12 finished requests, newest first
        new = {"prompt_tokens": 5000, "reused": 1000, "output_tokens": 300, "engine_generated": 300,
               "prompt_ms": 1500.0, "decode_ms": 4000.0, "decode_tok_s": 75.0, "prefill_tok_s": 2666.7}
        server.metrics_body = self._body(base, base)    # frame 1: the 12 requests are the whole history
        mod, frame1 = load_stats_module(server.server_address[1])
        server.metrics_body = self._body([new] + base[:11], [new] + base)   # frame 2: 13 seen, oldest dropped
        frame2 = mod.render()
        c1, c2 = rate_counts(frame1), rate_counts(frame2)
        for stdout in (frame1, frame2):
            for prefix in ("Prefill", "Generate"):
                self.assertIsNone(re.search(r"-\d", rate_line(stdout, prefix)), stdout)
        self.assertTrue(all(b >= a for a, b in zip(c1, c2)), f"counts decreased: {c1} -> {c2}")
        self.assertGreater(c2, c1)   # a real new request grew the running totals

    def test_maya_failed_json_frame_dashes_rates_without_jump(self):
        # three frames where the middle one cannot fetch /metrics: rates show "-" that frame, the cached and
        # output counts fall back to the running Prometheus mirror, and nothing goes negative, implausible or
        # discontinuous; the mirror's prompt_tokens_total is prompt minus reused, a different quantity than
        # Maya's prompt total, so only the prompt count dashes that frame; the good frames render real rates.
        server = self._maya_server()
        base = [self._req(k) for k in range(4)]
        fresh = sum(r["prompt_tokens"] - r["reused"] for r in base)
        reused = sum(r["reused"] for r in base)
        out = sum(r["output_tokens"] for r in base)
        prom = ('llamacpp:prompt_tokens_total{model="m"} %d\n'
                'llamacpp:prompt_tokens_cached_total{model="m"} %d\n'
                'llamacpp:prompt_seconds_total{model="m"} %s\n'
                'llamacpp:tokens_predicted_total{model="m"} %d\n'
                'llamacpp:tokens_predicted_seconds_total{model="m"} %s\n'
                % (fresh, reused, sum(r["prompt_ms"] for r in base) / 1000, out,
                   sum(r["decode_ms"] for r in base) / 1000))
        server.prometheus = prom
        server.metrics_body = self._body(base, base)
        mod, good1 = load_stats_module(server.server_address[1])
        server.metrics_body = None       # /metrics cannot be fetched this frame
        server.metrics_status = 404
        bad = mod.render()
        server.metrics_status = 200
        server.metrics_body = self._body(base, base)   # JSON is back
        good2 = mod.render()
        for stdout in (good1, bad, good2):
            for prefix in ("Prefill", "Generate"):
                line = rate_line(stdout, prefix)
                self.assertIsNone(re.search(r"-\d", line), stdout)   # never a negative
                try:
                    a = avg_tok_s(line)
                except ValueError:
                    continue   # avg is a dash: nothing to bound
                self.assertLess(0, a, line)
                self.assertLess(a, 100000, f"implausible avg in {line}")
        # the failed frame dashes both last and avg; the mirror keeps the cached and output counts continuous,
        # while the prompt count shows "-" rather than the mirror's different quantity (prompt minus reused)
        for prefix in ("Prefill", "Generate"):
            self.assertIn("last        - tok/s", rate_line(bad, prefix))
            self.assertRegex(rate_line(bad, prefix), r"avg +- tok/s")   # padded "avg" label, then the dash
        self.assertIn("(- tok", rate_line(bad, "Prefill"))
        self.assertEqual(rate_counts(bad), (None, reused, out))
        self.assertEqual(rate_counts(bad)[1:], rate_counts(good2)[1:])   # no jump in the mirrored counts
        self.assertEqual(rate_counts(good1)[0], sum(r["prompt_tokens"] for r in base))

    def test_maya_prefill_prompt_count_never_drops_when_a_multicall_finishes(self):
        # a multi-call request adds far more to reused than to prompt_tokens, so the old fresh count
        # (prompt minus reused) dropped by thousands when one finished; the Prefill count is Maya's running
        # prompt total, which only grows: a resumed request (prompt 17651, reused 50446, output 33512)
        # completing between two frames may not decrease the Prefill count or the Generate count.
        server = self._maya_server()
        base = [self._req(k) for k in range(3)]
        multi = {"prompt_tokens": 17651, "reused": 50446, "output_tokens": 33512,
                 "engine_generated": 716, "prompt_ms": 181.0, "decode_ms": 13137.2, "decode_tok_s": 54.5}
        server.metrics_body = self._body(base, base)
        mod, frame1 = load_stats_module(server.server_address[1])
        self.assertEqual(rate_counts(frame1)[0], sum(r["prompt_tokens"] for r in base))
        server.metrics_body = self._body([multi] + base, [multi] + base)   # the multi-call request finished
        frame2 = mod.render()
        c1, c2 = rate_counts(frame1), rate_counts(frame2)
        for stdout in (frame1, frame2):
            for prefix in ("Prefill", "Generate"):
                self.assertIsNone(re.search(r"-\d", rate_line(stdout, prefix)), stdout)
        self.assertTrue(all(b >= a for a, b in zip(c1, c2)), f"counts decreased: {c1} -> {c2}")
        self.assertEqual(c2, (c1[0] + 17651, c1[1] + 50446, c1[2] + 33512))   # the whole request, not a clamp

    def test_maya_footer_names_the_listed_requests_window(self):
        # Maya's avg covers only the requests its list holds, not the server's whole uptime
        out = run_stats(self.serve(metrics_body=MAYA_RESUMED))
        self.assertEqual(out.returncode, 0, out.stderr)
        footer = "last = newest request, avg(N) = over the N listed. Ctrl+C to exit."
        self.assertIn(footer, out.stdout)
        self.assertLessEqual(len(footer), 80)
        self.assertNotIn("since server start", out.stdout)

    def test_maya_multicall_newest_dashes_prefill_last_and_is_out_of_the_avg(self):
        # single-call rows have output_tokens == engine_generated, so output > 1.1 * generated + 16 marks a
        # visibly multi-call row, whose prompt_ms covers only its last call. The newest row here is multi-call
        # with reused < prompt (it would otherwise count), so Prefill last is "-" rather than the row below:
        # the footer promises the NEWEST request. The avg still uses the one consistent row, (2000-500) fresh /
        # 1.0 s = 1500.0, not the multi-call row's (10000-2000) / 1.0 s = 8000.0. Generate is untouched: its
        # last is still the newest row's decode_tok_s and both rows have consistent pairs for the avg.
        good = self._req(0)   # prompt 2000, reused 500, prompt_ms 1000 → prefill_tok_s 1500.0
        multi = {"prompt_tokens": 10000, "reused": 2000, "output_tokens": 9000, "engine_generated": 500,
                 "prompt_ms": 1000.0, "decode_ms": 5000.0, "decode_tok_s": 100.0, "prefill_tok_s": 8000.0}
        out = run_stats(self.serve(metrics_body={"engine": ENGINE, "live": {"state": "idle"},
                                                 "requests": [multi, good],
                                                 "totals": self._totals([multi, good])}))
        self.assertEqual(out.returncode, 0, out.stderr)
        pre = rate_line(out.stdout, "Prefill")
        self.assertIn("last        - tok/s", pre)   # never the multi-call row's 8000, nor the next row's 1500
        self.assertAlmostEqual(avg_tok_s(pre), 1500.0, delta=0.1)
        self.assertEqual(avg_req_count(pre), 1)       # the multi-call row is out of the avg window
        gen = rate_line(out.stdout, "Generate")
        self.assertIn("last    100.0 tok/s", gen)    # decode rates stay on the newest row
        self.assertEqual(avg_req_count(gen), 2)
        # a list of only multi-call rows leaves nothing consistent: Prefill last and avg both dash
        only = run_stats(self.serve(metrics_body={"engine": ENGINE, "live": {"state": "idle"},
                                                  "requests": [multi], "totals": self._totals([multi])}))
        self.assertEqual(only.returncode, 0, only.stderr)
        pre = rate_line(only.stdout, "Prefill")
        self.assertIn("last        - tok/s", pre)
        self.assertRegex(pre, r"avg +- tok/s")

    # the review row: a request of two roughly equal engine calls (output = 2 * engine_generated), which the
    # old output > 2 * generated + 16 test missed; 200 > 1.1 * 100 + 16 catches it
    TWO_CALLS = {"prompt_tokens": 5000, "reused": 1000, "output_tokens": 200, "engine_generated": 100,
                 "prompt_ms": 1000.0, "prefill_tok_s": 4000.0, "decode_ms": 1000.0, "decode_tok_s": 100.0}

    def _prefill_of(self, row):
        out = run_stats(self.serve(metrics_body={"engine": ENGINE, "live": {"state": "idle"},
                                                 "requests": [row], "totals": self._totals([row])}))
        self.assertEqual(out.returncode, 0, out.stderr)
        return rate_line(out.stdout, "Prefill")

    def assert_prefill_dashes(self, pre):
        self.assertIn("last        - tok/s", pre)     # newest row flagged: last shows "-"
        self.assertRegex(pre, r"avg +- tok/s")        # and nothing is left for the avg: no avg(N) at all

    def test_maya_two_equal_calls_row_is_multi_call(self):
        # output 200 vs engine_generated 100 (two equal calls): flagged multi-call, so its own
        # prefill_tok_s 4000.0 never reaches the dashboard, and it is out of the Prefill avg
        pre = self._prefill_of(self.TWO_CALLS)
        self.assert_prefill_dashes(pre)
        self.assertNotIn("4000.0", pre)

    def test_maya_unknown_generated_rows_are_out_of_prefill(self):
        # engine_generated missing, None or 0 with output 200 > 16 is "unknown": nothing ties the row's
        # prompt_ms to its whole prompt, so the row is treated like a multi-call one: Prefill last "-", avg "-"
        for drop in ("missing", None, 0):
            with self.subTest(engine_generated=drop):
                if drop == "missing":
                    row = {k: v for k, v in self.TWO_CALLS.items() if k != "engine_generated"}
                else:
                    row = {**self.TWO_CALLS, "engine_generated": drop}
                self.assert_prefill_dashes(self._prefill_of(row))

    def test_maya_single_call_row_close_to_generated_is_counted(self):
        # output 205 vs engine_generated 200: a single call with a slightly faster-growing output, 205 is not
        # over 1.1 * 200 + 16 = 236, so the row keeps its Prefill last and its avg(1) = (5000-1000) / 1.0 s
        row = {**self.TWO_CALLS, "output_tokens": 205, "engine_generated": 200}
        pre = self._prefill_of(row)
        self.assertRegex(pre, r"last\s+4000\.0 tok/s")
        self.assertAlmostEqual(avg_tok_s(pre), 4000.0, delta=0.1)
        self.assertEqual(avg_req_count(pre), 1)

    def test_maya_multi_call_flags(self):
        # the flag itself, beyond the two rows above: equal calls and any bigger output are multi-call,
        # missing/None/0 generated with output > 16 is unknown, a small or absent output is unjudgeable,
        # and output close to the generated stays a counted single call
        server = self._maya_server()
        server.metrics_body = self._body([], [])
        mod, _ = load_stats_module(server.server_address[1])
        row = lambda **kw: {**{"output_tokens": 200, "engine_generated": 100}, **kw}   # noqa: E731
        for r, want in [(row(), True),                                   # output == 2 * generated
                        (row(output_tokens=127), True),                  # just over 1.1 * 100 + 16 = 126
                        (row(output_tokens=126), False),                 # exactly at the bound: single call
                        (row(output_tokens=205, engine_generated=200), False),
                        (row(engine_generated=None), True),              # unknown: 200 > 16
                        (row(engine_generated=0), True),
                        ({**row(), "engine_generated": "100"}, True),    # a non-number generated is no proof
                        ({"output_tokens": 17}, True),                   # unknown even just past 16
                        ({"output_tokens": 16}, False),                  # a small output: not flagged
                        ({}, False),                                     # no output to compare: unjudgeable
                        ({"engine_generated": 100}, False)]:
            self.assertIs(mod.maya_multi_call(r), want, r)

    def test_maya_negative_totals_clamp_to_zero(self):
        # a malformed negative running total must render 0, never a negative count: the JSON totals clamp,
        # and so do the mirrored cached and output counts of a frame whose JSON fetch failed
        out = run_stats(self.serve(metrics_body={
            "engine": ENGINE, "live": {"state": "idle"}, "requests": [],
            "totals": {"requests": 1, "prompt_tokens": -5, "reused": -7, "output_tokens": -3}}))
        self.assertEqual(out.returncode, 0, out.stderr)
        for prefix in ("Prefill", "Generate"):
            self.assertIsNone(re.search(r"-\d", rate_line(out.stdout, prefix)), out.stdout)
        self.assertIn("(0 tok, 0 cached)", rate_line(out.stdout, "Prefill"))
        self.assertIn("(0 tok)", rate_line(out.stdout, "Generate"))
        self.assertEqual(rate_counts(out.stdout), (0, 0, 0))
        prom = ('llamacpp:prompt_tokens_total{model="m"} -4000\n'
                'llamacpp:prompt_tokens_cached_total{model="m"} -7\n'
                'llamacpp:tokens_predicted_total{model="m"} -9\n')
        out = run_stats(self.serve(metrics_status=404, prometheus=prom))   # JSON fetch failed
        self.assertEqual(out.returncode, 0, out.stderr)
        for prefix in ("Prefill", "Generate"):
            self.assertIsNone(re.search(r"-\d", rate_line(out.stdout, prefix)), out.stdout)
        self.assertEqual(rate_counts(out.stdout), (None, 0, 0))   # the prompt total still has no fallback

    def test_maya_avg_labels_padded_so_paren_columns_line_up(self):
        # "avg(12)" is wider than "avg" and grows again from N = 9 to 10, so Maya pads the label to a fixed
        # width: the "avg(N)" parens and the trailing count parens sit at the same index on the Prefill and
        # Generate lines for N = 2, 9 and 12, and for "-" too.
        server = self._maya_server()
        server.metrics_body = self._body([], [])
        mod, _ = load_stats_module(server.server_address[1])
        for n in (2, 9, 12):
            reqs = [self._req(k) for k in range(n)]
            server.metrics_body = self._body(reqs, reqs)
            frame = mod.render()
            pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
            self.assertEqual(avg_req_count(pre), n, pre)
            self.assertEqual(avg_req_count(gen), n, gen)
            self.assertEqual(pre.index("("), gen.index("("), f"N={n}:\n{pre}\n{gen}")   # the "avg(N)" column
            self.assertEqual(pre.rindex("("), gen.rindex("("), f"N={n}:\n{pre}\n{gen}")  # the count column
        server.metrics_body = self._body([], [])   # both avgs "-": no "avg(N)" label at all
        frame = mod.render()
        pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
        self.assertNotIn("avg(", pre)
        self.assertNotIn("avg(", gen)
        self.assertEqual(pre.rindex("("), gen.rindex("("), f"{pre}\n{gen}")
        # a count on one line and a dash on the other must not shift the trailing columns either
        keep = [self._req(k) for k in range(9)]
        nodec = [{k: v for k, v in r.items() if k not in ("decode_ms", "decode_tok_s")}
                 for r in keep]   # engine_generated kept (a missing one is "unknown", out of Prefill):
                                  # no decode_ms pair, so Prefill "avg(9)" and Generate "-"
        server.metrics_body = {"engine": ENGINE, "live": {"state": "idle"}, "requests": nodec,
                               "totals": self._totals(keep)}
        frame = mod.render()
        pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
        self.assertIn("avg(9)", pre)
        self.assertNotIn("avg(", gen)
        self.assertEqual(pre.rindex("("), gen.rindex("("), f"{pre}\n{gen}")

    def test_importing_the_module_does_not_start_the_polling_loop(self):
        # importing must parse no arguments, fetch nothing and print nothing, whatever sys.argv holds (even
        # the script's own --once args); rendering still works once configure() has applied parsed options
        import importlib.util
        from importlib.machinery import SourceFileLoader
        server = self._maya_server()
        server.metrics_body = self._body([], [])
        loader = SourceFileLoader("llm_stats_import_check", str(SCRIPT))
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("llm_stats_import_check", loader))
        argv, out = sys.argv, io.StringIO()
        sys.argv = ["llm-stats", "--once", "--engine", "maya", "--host", "127.0.0.1",
                    "--port", str(server.server_address[1])]
        try:
            with contextlib.redirect_stdout(out):
                loader.exec_module(mod)
        finally:
            sys.argv = argv
        self.assertEqual(server.hits, 0, "importing llm-stats reached the server")
        self.assertEqual(out.getvalue(), "", "importing llm-stats printed a dashboard")
        mod.configure(mod.p.parse_args(["--engine", "maya", "--host", "127.0.0.1",
                                        "--port", str(server.server_address[1])]))
        self.assertIn("Prefill", mod.render())   # rendering still works when asked for

    def test_importing_under_pytest_argv_does_not_exit(self):
        # argparse at module level made "import llm-stats" under pytest's own sys.argv exit(2) before the
        # tests ever ran: parsing belongs in main(), so exec_module must neither exit nor poll
        import importlib.util
        from importlib.machinery import SourceFileLoader
        server = self._maya_server()
        server.metrics_body = self._body([], [])
        loader = SourceFileLoader("llm_stats_pytest_argv", str(SCRIPT))
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("llm_stats_pytest_argv", loader))
        argv, out = sys.argv, io.StringIO()
        sys.argv = ["pytest", "-q"]
        try:
            with contextlib.redirect_stdout(out):
                loader.exec_module(mod)   # SystemExit here (or any hit on the stub) fails the test
        finally:
            sys.argv = argv
        self.assertEqual(server.hits, 0, "importing llm-stats reached the server")
        self.assertEqual(out.getvalue(), "", "importing llm-stats printed a dashboard")
        mod.configure(mod.p.parse_args(["--engine", "maya", "--host", "127.0.0.1",
                                        "--port", str(server.server_address[1])]))
        self.assertIn("Prefill", mod.render())

    def test_maya_rate_lines_and_footer_fit_80_columns(self):
        # 8-digit running totals, with and without a live request: Prefill, Generate and the footer all stay
        # within 80 columns. The counts compact to k/M ("12.3M, 87.7M cached") and a live request's progress
        # alone is shown when the totals would not fit beside it ("+12M/88M in flight").
        server = self._maya_server()
        reqs = [self._req(k) for k in range(3)]
        big = {"prompt_tokens": 12345678, "reused": 87654321, "output_tokens": 98765432}
        server.metrics_body = {"engine": ENGINE, "live": {"state": "idle"}, "requests": reqs, "totals": big}
        mod, _ = load_stats_module(server.server_address[1])
        cases = [
            ({"state": "idle"}, "(12.3M, 87.7M cached)", "(98765432 tok)"),   # exact counts still fit Generate
            ({"state": "reading", "prefill_tok_s_mean": 1234.5, "prompt_reused": 10, "prompt_read": 12345678,
              "prompt_total": 87654321, "prompt_tokens": 87654321},           # 8-digit read of an 8-digit prompt
             "(+12M/88M in flight)", "(98765432 tok)"),
            ({"state": "generating", "tok_s": 31.2, "generated": 12345678},
             "(12.3M, 87.7M cached)", "(98.8M, +12M in flight)"),
        ]
        for live, want_pre, want_gen in cases:
            server.metrics_body = {"engine": ENGINE, "live": live, "requests": reqs, "totals": big}
            frame = mod.render()
            pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
            self.assertIn(want_pre, pre)     # the compact form is what rendered, not a wider one
            self.assertIn(want_gen, gen)
            self.assertLessEqual(len(pre), 80, pre)
            self.assertLessEqual(len(gen), 80, gen)
            self.assertLessEqual(len(frame.splitlines()[-1]), 80, frame)   # the footer
            self.assertEqual(pre.rindex("("), gen.rindex("("), f"{pre}\n{gen}")
        # the widest case that still keeps exact counts: two 6-digit ones, "(999949, 987654 cached)", which
        # fills line 80 exactly; the first 7-digit count is where the k/M compaction takes over
        server.metrics_body = {"engine": ENGINE, "live": {"state": "idle"}, "requests": reqs,
                               "totals": {"prompt_tokens": 999949, "reused": 987654, "output_tokens": 1234}}
        pre = rate_line(mod.render(), "Prefill")
        self.assertIn("(999949, 987654 cached)", pre)
        self.assertEqual(len(pre), 80, pre)
        server.metrics_body = {"engine": ENGINE, "live": {"state": "idle"}, "requests": reqs,
                               "totals": {"prompt_tokens": 9999999, "reused": 9876543, "output_tokens": 1234}}
        pre = rate_line(mod.render(), "Prefill")
        self.assertIn("(10.0M, 9.9M cached)", pre)
        self.assertLessEqual(len(pre), 80, pre)
        # maya_count stays within 6 characters whatever the count, which is what keeps the lines within 80
        self.assertEqual(max(len(mod.maya_count(v)) for v in
                             (0, 999, 1000, 48987, 999949, 999999, 12345678, 987654321,
                              10 ** 12, 10 ** 15, 10 ** 18 - 1)), 6)
        self.assertEqual(mod.maya_count(999999), "1.0M")   # never "1000.0k"
        # the widest live prefill: 9-digit read and total still fit, with "(" aligned with Generate's
        live = {"state": "reading", "prefill_tok_s_mean": 99999.9, "prompt_reused": 0, "prompt_read": 999999997,
                "prompt_total": 999999999, "prompt_tokens": 999999999}
        server.metrics_body = {"engine": ENGINE, "live": live, "requests": reqs, "totals": big}
        frame = mod.render()
        pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
        self.assertIn("(+1.0B/1.0B in flight)", pre)
        self.assertLessEqual(len(pre), 80, pre)
        self.assertEqual(pre.rindex("("), gen.rindex("("), f"{pre}\n{gen}")
        live = {**live, "prompt_read": 998999999, "prompt_total": 999499999, "prompt_tokens": 999499999}
        server.metrics_body = {"engine": ENGINE, "live": live, "requests": reqs, "totals": big}
        frame = mod.render()
        pre, gen = rate_line(frame, "Prefill"), rate_line(frame, "Generate")
        self.assertIn("(+999M/999M in flight)", pre)   # the widest count pair below a billion
        self.assertLessEqual(len(pre), 80, pre)
        self.assertEqual(pre.rindex("("), gen.rindex("("), f"{pre}\n{gen}")
        # maya_short stays within 4 characters for every count below a billion
        self.assertEqual(max(len(mod.maya_short(v)) for v in
                             (0, 999, 1000, 9949, 9950, 99999, 999499, 999500, 999999, 9949999, 999499999,
                              999500000, 999999999)), 4)
        self.assertEqual(mod.maya_short(999999), "1.0M")   # never "1000k"
        self.assertEqual(mod.maya_short(12345), "12k")


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


class BasaltLiveServer(HTTPServer):
    live = {}
    slots = [{"id": 0, "n_ctx": 524288, "is_processing": True, "ctx_used": 1000, "tok_s": 0.0}]


class BasaltLiveHandler(StubHandler):
    def do_GET(self):
        self.respond({
            "/health": {"service": "basalt", "status": "ok"},
            "/v1/models": {"data": [{"id": "basalt-test", "meta": {"n_ctx": 524288}}]},
            "/metrics": {"engine": BASALT_ENGINE, "live": self.server.live,
                         "totals": {"requests": 0, "prompt": 0, "reused": 0, "written": 0,
                                    "prompt_ms": 0.0, "decode_ms": 0.0}},
            "/slots": self.server.slots,
        })


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

    def test_footer_unchanged_for_basalt_and_llamacpp(self):
        # only Maya's footer changed: the others still claim the avg spans the server's whole uptime
        for handler in (BasaltHandler, LlamaCppHandler):
            server = serve(handler)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            out = run_stats(server.server_address[1])
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("last = most recent request, avg = since server start. Ctrl+C to exit.", out.stdout)
            self.assertNotIn("Maya lists", out.stdout)


class LlmStatsBasaltLiveTest(unittest.TestCase):
    """Basalt's live block: a Live line and "now" rates while its first request still runs (totals still 0).
    Field shapes follow Basalt's serve/service.py: phase "reading the prompt" while reading, a slot's `reused`
    stays None until the first token, and `prefill_done` counts the cached prefix as read."""
    GEN = {"state": "generating", "phase": "thinking", "prompt_tokens": 23499, "generated": 10403,
           "elapsed_s": 129.9, "tok_s": 104.7, "prefill_tok_s_mean": 1797.9, "slots": {"active": 1}, "queued": 0}
    READ = {"state": "reading", "phase": "reading the prompt", "prompt_tokens": 23499, "generated": 0,
            "elapsed_s": 4.2, "tok_s": None, "prefill_tok_s_mean": 1800.5, "slots": {"active": 1}, "queued": 0}
    GEN_ROW = {"id": 0, "n_ctx": 524288, "is_processing": True, "state": "decoding", "max_tokens": 64000,
               "generated": 10403, "prompt_tokens": 23499, "prefill_done": 23499, "reused": 0}
    READ_ROW = {"id": 0, "n_ctx": 524288, "is_processing": True, "state": "prefill", "generated": 0,
                "prompt_tokens": 23499, "prefill_done": 8192, "reused": None}

    def run_live(self, live, slots=None):
        server = BasaltLiveServer(("127.0.0.1", 0), BasaltLiveHandler)
        server.live = live
        if slots is not None:
            server.slots = slots
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        out = run_stats(server.server_address[1])
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_generating_live_line_and_now_rate(self):
        out = self.run_live(self.GEN, [self.GEN_ROW])
        self.assertEqual(live_line(out), "Live       thinking     104.7 tok/s  elapsed  129.9 s"
                                         "  generated 10403/64000 tok")
        self.assertEqual(rate_line(out, "Generate"),
                         "Generate   now     104.7 tok/s   avg        - tok/s   (0 tok, +10403 in flight)")
        self.assertTrue(rate_line(out, "Prefill").startswith("Prefill    last        - tok/s"))

    def test_reading_live_line_shortens_phase(self):
        line = live_line(self.run_live(self.READ, [self.READ_ROW]))   # "reading the prompt" -> "reading"
        self.assertEqual(line, "Live       reading" + " " * 17 + "  elapsed    4.2 s  prompt 8192/23499 tok")

    def test_long_phases_get_short_ascii_labels_and_columns_hold(self):
        gen = live_line(self.run_live(self.GEN, [self.GEN_ROW]))
        for phase, want in [("writing a tool call: search_files", "tool call"), ("tool call complete", "tool done"),
                            ("summarizing context", "summarizi~")]:
            line = live_line(self.run_live({**self.GEN, "phase": phase}, [self.GEN_ROW]))
            self.assertTrue(line.startswith(f"Live       {want:<10}  "), line)
            self.assertEqual(gen.index("elapsed"), line.index("elapsed"))
            line.encode("ascii")   # a non-ASCII cut mark would crash a non-UTF-8 terminal
        self.assertLessEqual(len(gen), 80)

    def test_newest_request_without_a_slot_does_not_label_the_running_one(self):
        # parallel: the live block is a newer request still waiting for a slot, while slot 0 runs another one
        live = {**self.GEN, "phase": "reading the prompt", "prompt_tokens": 500, "generated": 0, "elapsed_s": 0.3}
        out = self.run_live(live, [self.GEN_ROW])
        line = live_line(out)
        self.assertTrue(line.endswith("generated 10403/64000 tok"), line)
        self.assertNotIn("reading", line)
        self.assertNotIn("elapsed", line)
        self.assertIn("+10403 in flight", rate_line(out, "Generate"))

    def test_reading_prefill_counts_cache_until_basalt_reports_it(self):
        out = self.run_live(self.READ, [self.READ_ROW])   # reused None: prefill_done includes the cached prefix
        self.assertEqual(rate_line(out, "Prefill"), "Prefill    now    1800.5 tok/s   avg        - tok/s"
                                                    "   (0 tok, 0 cached, 8192/23499 read incl. cached)")
        self.assertTrue(rate_line(out, "Generate").startswith("Generate   last        - tok/s"))

    def test_reading_known_reuse_counts_only_fresh_tokens(self):
        out = self.run_live(self.READ, [{**self.READ_ROW, "prefill_done": 9000, "reused": 8000}])
        self.assertIn("+1000/15499 in flight", rate_line(out, "Prefill"))

    def test_reading_before_progress_hides_stale_rate(self):
        for done in (None, 0):   # the batched engine still holds the previous request's prefill rate
            out = self.run_live(self.READ, [{**self.READ_ROW, "prefill_done": done}])
            self.assertTrue(rate_line(out, "Prefill").startswith("Prefill    last "), (done, out))
            self.assertTrue(live_line(out).endswith("prompt 23499 tok" if done is None else "prompt 0/23499 tok"), out)

    def test_several_busy_slots_show_only_totals(self):
        live = {**self.GEN, "tok_s": 210.0, "generated": 0, "phase": "reading the prompt", "elapsed_s": 0.5}
        rows = [{**self.GEN_ROW, "generated": 5000}, {**self.READ_ROW, "id": 1, "generated": 0},
                {"id": 2, "is_processing": False}]
        out = self.run_live(live, rows)
        line = live_line(out)
        self.assertTrue(line.endswith("2 requests, generated 5000 tok"), line)
        self.assertNotIn("reading", line)
        self.assertNotIn("elapsed", line)
        self.assertIn("210.0 tok/s", line)
        self.assertEqual(rate_line(out, "Generate"),
                         "Generate   now     210.0 tok/s   avg        - tok/s   (0 tok, +5000 in flight)")
        self.assertTrue(rate_line(out, "Prefill").startswith("Prefill    last "))

    def test_one_busy_of_several_slots_uses_its_row(self):
        out = self.run_live(self.GEN, [{"id": 0, "is_processing": False}, {**self.GEN_ROW, "id": 1}])
        self.assertTrue(live_line(out).endswith("generated 10403/64000 tok"))

    def test_bad_values_fall_back(self):
        for live in [{**self.GEN, "tok_s": "fast", "generated": True, "elapsed_s": None},
                     {**self.GEN, "tok_s": -1.0}, {**self.READ, "prefill_tok_s_mean": 0}, "broken", []]:
            out = self.run_live(live, [self.READ_ROW])
            self.assertTrue(rate_line(out, "Prefill").startswith("Prefill    last "), (live, out))
            self.assertTrue(rate_line(out, "Generate").startswith("Generate   last "), (live, out))
            self.assertNotIn("in flight", out)

    def test_idle_has_no_live_line(self):
        out = self.run_live({"state": "idle", "tok_s": None})
        self.assertNotIn("Live", out)
        self.assertTrue(rate_line(out, "Generate").startswith("Generate   last "))


def slot_line(stdout):
    return next((l for l in stdout.splitlines() if l.startswith("Slot 0")), "")


class LlmStatsMayaCtxTest(unittest.TestCase):
    """Maya's used context on the slot line, from /metrics (its /slots has no used count)."""
    serve = LlmStatsMayaTest.serve

    def slot(self, **cfg):
        out = run_stats(self.serve(**cfg))
        self.assertEqual(out.returncode, 0, out.stderr)
        return slot_line(out.stdout)

    def test_generating_counts_prompt_plus_generated(self):
        line = self.slot(live={**LIVE, "prompt_tokens": 1000})
        self.assertIn("ctx    7905/524288", line)
        self.assertIn("[", line)   # the bar is drawn

    def test_reading_counts_generated_of_a_resumed_request(self):
        line = self.slot(live={"state": "reading", "phase": "reading the prompt", "prompt_tokens": 5000,
                               "generated": 6905})
        self.assertIn("ctx   11905/524288", line)

    def test_reading_fresh_request_counts_prompt(self):
        line = self.slot(live={"state": "reading", "prompt_tokens": 5000, "generated": 0})
        self.assertIn("ctx    5000/524288", line)

    def test_unloaded_shows_dash(self):
        line = self.slot(live={"state": "unloaded"}, requests=[{"prompt_tokens": 100, "output_tokens": 5}])
        self.assertIn("ctx       -/524288", line)

    def test_bad_newest_request_does_not_fall_back(self):
        line = self.slot(live={"state": "idle"}, requests=["bad", {"prompt_tokens": 100, "output_tokens": 5}])
        self.assertIn("ctx       -/524288", line)

    def test_idle_uses_last_request(self):
        line = self.slot(live={"state": "idle"}, requests=[
            {"prompt_tokens": 229237, "output_tokens": 1307}, {"prompt_tokens": 10, "output_tokens": 5}])
        self.assertIn("ctx  230544/524288", line)

    def test_idle_without_requests_shows_dash(self):
        line = self.slot(live={"state": "idle"})
        self.assertIn("ctx       -/524288", line)
        self.assertNotIn("[", line)

    def test_running_without_prompt_tokens_shows_dash(self):
        self.assertIn("ctx       -/524288", self.slot(live=LIVE))

    def test_bad_values_show_dash(self):
        for live, reqs in [({**LIVE, "prompt_tokens": True}, []),
                           ({**LIVE, "prompt_tokens": -3}, []),
                           ({"state": "idle"}, [{"prompt_tokens": "9", "output_tokens": 1}]),
                           ({"state": "idle"}, [{"prompt_tokens": 9}]),
                           ({"state": "idle"}, ["not a dict"])]:
            with self.subTest(live=live, reqs=reqs):
                self.assertIn("ctx       -/524288", self.slot(live=live, requests=reqs))

    def test_bad_generated_counts_prompt_only(self):
        self.assertIn("ctx    1000/524288", self.slot(live={**LIVE, "prompt_tokens": 1000, "generated": -1}))

    def test_non_list_requests_body_renders(self):
        self.assertIn("ctx       -/524288", self.slot(live={"state": "idle"}, requests={"a": 1}))

    def test_used_clamps_to_context(self):
        self.assertIn("ctx  524288/524288", self.slot(live={**LIVE, "prompt_tokens": 600000}))


NVIDIA_GPU_STUB = r"""#!/bin/sh
case "$*" in
  *compute-apps*) [ -f "__APPS__" ] || exit 1; cat "__APPS__"; exit 0;;
  *uuid*)         [ -f "__UUIDS__" ] || exit 1; cat "__UUIDS__"; exit 0;;
  *utilization*)  [ -f "__GPUS__" ] || exit 1; cat "__GPUS__"; exit 0;;
esac
exit 1
"""

SS_STUB = r"""#!/bin/sh
[ -f "__SS__" ] || exit 1
cat "__SS__"
"""


class LlmStatsGpuTest(unittest.TestCase):
    """The dashboard's GPU lines on a two-card box: the cards the server on --port actually holds.
    Stub nvidia-smi and ss sit first on PATH; /proc is a fake tree under LLM_STATS_PROC."""
    GPUS_CSV = ("0, 10, 1234, 97887, 60.5, 41\n"
                "1, 99, 89498, 97887, 300.2, 61\n")
    UUIDS_CSV = "0, GPU-AAAA\n1, GPU-BBBB\n"
    CARD1 = "GPU 1  99%  VRAM  87.4/95.6 GiB    300 W  61 C"
    CARD0 = "GPU 0  10%  VRAM   1.2/95.6 GiB     60 W  41 C"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="llm-stats-gpu-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stub = self.tmp / "stub"
        self.stub.mkdir()
        self.apps = self.stub / "apps.csv"
        self.gpus = self.stub / "gpus.csv"
        self.uuids = self.stub / "uuids.csv"
        self.ssout = self.stub / "ss.out"
        nvidia = self.stub / "nvidia-smi"
        nvidia.write_text(NVIDIA_GPU_STUB.replace("__APPS__", str(self.apps))
                                   .replace("__UUIDS__", str(self.uuids))
                                   .replace("__GPUS__", str(self.gpus)))
        nvidia.chmod(0o755)
        ss = self.stub / "ss"
        ss.write_text(SS_STUB.replace("__SS__", str(self.ssout)))
        ss.chmod(0o755)
        self.gpus.write_text(self.GPUS_CSV)
        self.uuids.write_text(self.UUIDS_CSV)
        self.proc = self.tmp / "proc"

    def fake_proc(self, procs):
        """A fake /proc tree: procs maps pid -> (ppid, session). The comm field holds spaces and parens,
        so the parser has to cut at the last ")" like the real /proc one does."""
        shutil.rmtree(self.proc, ignore_errors=True)
        for pid, (ppid, sid) in procs.items():
            d = self.proc / str(pid)
            d.mkdir(parents=True)
            (d / "stat").write_text(f"{pid} (fake (weird) server) S {ppid} {ppid} {sid} 0 0\n")

    def listen(self, pid):
        self.ssout.write_text(f'LISTEN 0 128 127.0.0.1:9 0.0.0.0:* users:(("llama-server",pid={pid},fd=6))\n')

    def serve(self):
        server = HTTPServer(("127.0.0.1", 0), LlamaCppHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def run_dash(self, port, *args):
        env = {**ENV, "PATH": f"{self.stub}{os.pathsep}{os.environ['PATH']}",
               "LLM_STATS_PROC": str(self.proc)}
        return subprocess.run([sys.executable, str(SCRIPT), "--once", "--host", "127.0.0.1",
                               "--port", str(port), *args], capture_output=True, text=True,
                              timeout=30, env=env)

    def gpu_lines(self, stdout):
        return [ln for ln in stdout.splitlines() if ln.startswith("GPU")]

    def test_server_holding_card_one_shows_only_card_one(self):
        self.fake_proc({4000: (1, 4000), 4001: (4000, 4000)})   # the listener leads the session
        self.apps.write_text("4001, GPU-BBBB\n")
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout), [self.CARD1])
        self.assertNotIn("GPU 0", out.stdout)

    def test_server_holding_both_cards_shows_both(self):
        self.fake_proc({4000: (1, 4000), 4001: (4000, 4000)})
        self.apps.write_text("4001, GPU-BBBB\n4000, GPU-AAAA\n")
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([ln.split()[1] for ln in self.gpu_lines(out.stdout)], ["0", "1"])

    def test_holder_found_through_ppid_when_listener_is_not_the_leader(self):
        # the listener 4200 belongs to session 4000 led by 4100, which holds GPU 0: only the listener's own
        # subtree counts, so the session mate's card stays hidden and the listener's child's card shows
        self.fake_proc({4100: (1, 4000), 4200: (4100, 4000), 4201: (4200, 4000)})
        self.apps.write_text("4100, GPU-AAAA\n4201, GPU-BBBB\n")
        port = self.serve()
        self.listen(4200)
        out = self.run_dash(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout), [self.CARD1])

    def test_no_holding_process_falls_back_to_every_card(self):
        self.fake_proc({4000: (1, 4000)})
        self.apps.write_text("9999, GPU-BBBB\n")   # a compute app that is not this server
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([ln.split()[1] for ln in self.gpu_lines(out.stdout)], ["0", "1"])

    def test_gpus_all_shows_every_card_whatever_the_holder(self):
        self.fake_proc({4000: (1, 4000)})
        self.apps.write_text("4000, GPU-BBBB\n")
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port, "--gpus", "all")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([ln.split()[1] for ln in self.gpu_lines(out.stdout)], ["0", "1"])

    def test_gpus_list_shows_only_that_card(self):
        self.fake_proc({4000: (1, 4000)})
        self.apps.write_text("4000, GPU-BBBB\n")
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port, "--gpus", "0")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([ln.split()[1] for ln in self.gpu_lines(out.stdout)], ["0"])
        out = self.run_dash(port, "--gpus", "0,1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([ln.split()[1] for ln in self.gpu_lines(out.stdout)], ["0", "1"])

    def test_nvidia_smi_failure_shows_na(self):
        self.fake_proc({4000: (1, 4000)})
        self.apps.write_text("4000, GPU-BBBB\n")
        self.gpus.unlink()   # the stub exits 1 for the stats query, like a failing nvidia-smi
        port = self.serve()
        self.listen(4000)
        out = self.run_dash(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout), ["GPU n/a"])

    def test_na_power_row_shows_dash_and_keeps_the_other_card(self):
        # a row with "[N/A]" power is parsed on its own: its dash must not hide the card whose row is fine
        self.gpus.write_text("0, 10, 1234, 97887, 60.5, 41\n"
                             "1, 99, 89498, 97887, [N/A], 61\n")
        port = self.serve()
        out = self.run_dash(port, "--gpus", "0")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout), [self.CARD0])   # the healthy card prints as it always did
        out = self.run_dash(port, "--gpus", "all")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout),
                         [self.CARD0, "GPU 1  99%  VRAM  87.4/95.6 GiB      - W  61 C"])

    def test_only_unusable_rows_shows_na(self):
        # every row skipped: wrong field count, or an index that is not an integer
        self.gpus.write_text("garbage\nbroken, row\nx, 1, 2, 3, 4, 5\n")
        port = self.serve()
        out = self.run_dash(port, "--gpus", "all")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(self.gpu_lines(out.stdout), ["GPU n/a"])

    def test_bad_gpus_value_exits_with_an_argparse_error(self):
        port = self.serve()
        out = self.run_dash(port, "--gpus", "x")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("--gpus", out.stderr)


MULTI_NVIDIA_STUB = r"""#!/bin/sh
case "$*" in
  *utilization*) cat "__GPUS__"; exit 0;;
esac
exit 1
"""


class LlmStatsMultiPortTest(unittest.TestCase):
    """Several --port targets on one screen: one block per port in the order given, one blank line
    between the blocks, the footer once at the end. --gpus all plus a stub nvidia-smi printing a
    fixed CSV keep the real machine's GPU output out of every assertion."""
    GPUS_CSV = ("0, 10, 1234, 97887, 60.5, 41\n"
                "1, 99, 89498, 97887, 300.2, 61\n")
    FOOTER = "last = most recent request, avg = since server start. Ctrl+C to exit."
    MAYA_FOOTER = "last = newest request, avg(N) = over the N listed. Ctrl+C to exit."
    CLEAR = "\033[H\033[2J"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="llm-stats-multi-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stub = self.tmp / "stub"
        self.stub.mkdir()
        gpus = self.stub / "gpus.csv"
        gpus.write_text(self.GPUS_CSV)
        nvidia = self.stub / "nvidia-smi"
        nvidia.write_text(MULTI_NVIDIA_STUB.replace("__GPUS__", str(gpus)))
        nvidia.chmod(0o755)
        self.env = {**ENV, "PATH": f"{self.stub}{os.pathsep}{os.environ['PATH']}"}
        old_path = os.environ["PATH"]
        os.environ["PATH"] = self.env["PATH"]   # the in-process render() below reaches nvidia-smi too
        self.addCleanup(os.environ.__setitem__, "PATH", old_path)

    def serve(self):
        server = serve(LlamaCppHandler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    @staticmethod
    def dead_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def run_stats(self, *ports):
        args = [sys.executable, str(SCRIPT), "--once", "--host", "127.0.0.1", "--gpus", "all"]
        for port in ports:
            args += ["--port", str(port)]
        return subprocess.run(args, capture_output=True, text=True, timeout=30, env=self.env)

    def test_two_ports_render_one_block_each_in_the_order_given(self):
        pa, pb = self.serve(), self.serve()
        out = self.run_stats(pb, pa)   # the order given on the command line, whatever the port numbers
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        ib = next(i for i, ln in enumerate(lines) if f"http://127.0.0.1:{pb}   model" in ln)
        ia = next(i for i, ln in enumerate(lines) if f"http://127.0.0.1:{pa}   model" in ln)
        self.assertLess(ib, ia)
        self.assertEqual(ib, 0)                          # the first block starts the output
        self.assertEqual(lines[ia - 1], "")              # one blank line separates the blocks
        self.assertNotEqual(lines[ia - 2], "")           # ... and only one
        self.assertEqual(len([ln for ln in lines if ln.startswith("Prefill")]), 2)
        self.assertEqual(out.stdout.count("GPU 0 "), 2)   # every block keeps its own GPU lines
        self.assertEqual(out.stdout.count("Ctrl+C to exit."), 1)   # the footer once, at the very end
        self.assertEqual(lines[-1], self.FOOTER)

    def test_unreachable_port_gets_its_own_block(self):
        port, dead = self.serve(), self.dead_port()
        for first, second in ((dead, port), (port, dead)):   # the dead port first, and last
            out = self.run_stats(first, second)
            self.assertEqual(out.returncode, 0, out.stderr)
            lines = out.stdout.splitlines()
            at = {v: next(i for i, ln in enumerate(lines)
                          if ln.startswith(f"Cannot reach http://127.0.0.1:{v}")
                          or f"http://127.0.0.1:{v}   model" in ln)
                  for v in (port, dead)}
            self.assertLess(at[first], at[second])           # blocks keep the order given
            self.assertIn(f"Cannot reach http://127.0.0.1:{dead}", out.stdout)
            self.assertIn(f"http://127.0.0.1:{port}   model", out.stdout)   # the other keeps rendering
            self.assertEqual(out.stdout.count("Ctrl+C to exit."), 1)
            self.assertEqual(lines[-1], self.FOOTER)

    def test_repeated_port_renders_once(self):
        port = self.serve()
        twice = self.run_stats(port, port)
        once = self.run_stats(port)
        self.assertEqual(twice.returncode, 0, twice.stderr)
        self.assertEqual(once.stdout.count("Prefill"), 1)
        self.assertEqual(twice.stdout, once.stdout)   # deduplicated, the same single block

    def test_single_port_output_is_render_output(self):
        # one --port: byte-identical to what render() (the pre-multi-port path) prints directly
        port = self.serve()
        out = self.run_stats(port)
        self.assertEqual(out.returncode, 0, out.stderr)
        import importlib.util
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader("llm_stats_multi_single", str(SCRIPT))
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("llm_stats_multi_single", loader))
        loader.exec_module(mod)
        mod.configure(mod.p.parse_args(["--host", "127.0.0.1", "--gpus", "all", "--port", str(port)]))
        self.assertEqual(out.stdout, mod.render() + "\n")

    def test_live_mode_clears_once_per_frame_for_the_whole_set(self):
        pa, pb = self.serve(), self.serve()
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "--host", "127.0.0.1",
                                 "--gpus", "all", "--interval", "0.2",
                                 "--port", str(pa), "--port", str(pb)],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, env=self.env)
        try:
            data = ""
            deadline = time.monotonic() + 30
            while data.count(self.CLEAR) < 2 and time.monotonic() < deadline:
                data += proc.stdout.read(4096)
        finally:
            proc.kill()
            proc.wait(timeout=10)
        self.assertGreaterEqual(data.count(self.CLEAR), 2, "no second live frame arrived")
        first, _, rest = data.partition(self.CLEAR)
        frame1, _, _ = rest.partition(self.CLEAR)
        self.assertEqual(first, "")   # a frame starts with the clear, not with a block
        self.assertNotIn(self.CLEAR, frame1)   # one clear per frame, not one per block
        self.assertEqual(frame1.count("Ctrl+C to exit."), 1)
        self.assertIn(f"http://127.0.0.1:{pa}   model", frame1)
        self.assertIn(f"http://127.0.0.1:{pb}   model", frame1)

    def serve_handler(self, handler, cls=HTTPServer):
        server = cls(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def load_frames_module(self, *ports):
        """The module in-process, frame() over every given port (--gpus all: the stub nvidia-smi answers)."""
        import importlib.util
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader("llm_stats_footer_frames", str(SCRIPT))
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("llm_stats_footer_frames", loader))
        loader.exec_module(mod)
        args = []
        for port in ports:
            args += ["--port", str(port)]
        mod.configure(mod.p.parse_args(["--host", "127.0.0.1", "--gpus", "all", *args]))
        return mod

    def test_frame_prints_every_distinct_footer_once(self):
        # mixed engines on one screen: each engine's footer once, in the order first seen, at the end
        basalt = self.serve_handler(BasaltHandler)
        maya = self.serve_handler(MayaHandler, MayaServer)
        frame = self.load_frames_module(basalt, maya).frame()
        self.assertEqual(frame.splitlines()[-2:], [self.FOOTER, self.MAYA_FOOTER])
        self.assertEqual(frame.count(self.FOOTER), 1)   # the block itself no longer carries it
        self.assertEqual(frame.count(self.MAYA_FOOTER), 1)
        frame = self.load_frames_module(maya, basalt).frame()   # order follows the blocks, not the engine
        self.assertEqual(frame.splitlines()[-2:], [self.MAYA_FOOTER, self.FOOTER])
        self.assertEqual(frame.count("Ctrl+C to exit."), 2)

    def test_frame_prints_one_footer_for_two_same_engine_servers(self):
        frame = self.load_frames_module(self.serve_handler(BasaltHandler),
                                        self.serve_handler(BasaltHandler)).frame()
        self.assertIn("Basalt @", frame)
        self.assertEqual(frame.count("Ctrl+C to exit."), 1)
        self.assertEqual(frame.splitlines()[-1], self.FOOTER)


if __name__ == "__main__":
    unittest.main()
