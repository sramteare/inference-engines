"""llm-serve refuses to launch while another server still holds GPU memory."""
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "bin" / "llm-serve"
MARKER = "STUB-LLAMA-SERVER-LAUNCHED"
WAIT_MSG = "Waiting for GPU memory"

NVIDIA_STUB = r"""#!/bin/sh
data="__DATA__"
[ -f "$data" ] || exit 1
while IFS= read -r line; do
  [ -n "$line" ] || continue
  pid="${line%%,*}"
  kill -0 "$pid" 2>/dev/null && printf '%s\n' "$line"
done < "$data"
exit 0
"""

# Answers --query-gpu from the holder's state (held: 2000 MiB free, gone: 90000),
# and --query-compute-apps from the data file, filtered by liveness.
FLEX_NVIDIA_STUB = r"""#!/bin/sh
holder="__HOLDER__"
case "$*" in
  *--query-gpu*)
    if kill -0 "$holder" 2>/dev/null; then echo 2000; else echo 90000; fi
    exit 0
    ;;
esac
data="__DATA__"
[ -f "$data" ] || exit 1
while IFS= read -r line; do
  [ -n "$line" ] || continue
  pid="${line%%,*}"
  kill -0 "$pid" 2>/dev/null && printf '%s\n' "$line"
done < "$data"
exit 0
"""

# Ignores SIGTERM (ignored dispositions survive exec), so only the
# timeout --kill-after can end it.
TERM_NVIDIA_STUB = r"""#!/bin/sh
trap '' TERM
exec sleep 30
"""

# Reports normally, fails (exit 1) on the second call, then reports again.
FAIL_MID_NVIDIA_STUB = r"""#!/bin/sh
counter="__COUNTER__"
n=$(cat "$counter" 2>/dev/null || echo 0)
n=$((n + 1))
printf '%s\n' "$n" > "$counter"
[ "$n" -eq 2 ] && exit 1
data="__DATA__"
[ -f "$data" ] || exit 1
while IFS= read -r line; do
  [ -n "$line" ] || continue
  pid="${line%%,*}"
  kill -0 "$pid" 2>/dev/null && printf '%s\n' "$line"
done < "$data"
exit 0
"""

# Reports normally, but on the second call hangs for 30 s (execs sleep), so a
# poll during the wait would block unless its timeout is capped by the deadline.
SLOW_SECOND_NVIDIA_STUB = r"""#!/bin/sh
counter="__COUNTER__"
n=$(cat "$counter" 2>/dev/null || echo 0)
n=$((n + 1))
printf '%s\n' "$n" > "$counter"
[ "$n" -eq 2 ] && exec sleep 30
data="__DATA__"
[ -f "$data" ] || exit 1
while IFS= read -r line; do
  [ -n "$line" ] || continue
  pid="${line%%,*}"
  kill -0 "$pid" 2>/dev/null && printf '%s\n' "$line"
done < "$data"
exit 0
"""


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlmServeGpuWaitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="llm-serve-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stub = self.tmp / "stub"
        self.stub.mkdir()
        self.models = self.tmp / "models"
        self.models.mkdir()
        self.logs = self.tmp / "logs"
        self.data = self.stub / "nvidia.csv"

        # Real ELF copied (under the name llama-server) to build holders so that
        # /proc/<pid>/exe resolves to a file called llama-server, exactly like a
        # genuine server. bash is a standalone ELF (coreutils' sleep is a
        # multicall keyed on argv[0] and would exit when renamed).
        shell = shutil.which("bash")
        if shell is None:
            self.skipTest("bash not found on this machine")
        self.holder_shell = shell
        # A PROC_ROOT that never exists: readlink $PROC_ROOT/<pid>/exe fails, so
        # gpu_holders falls back to matching the reported name.
        self.no_proc = self.tmp / "no-proc"

        self.nvidia = self.stub / "nvidia-smi"
        self.nvidia.write_text(NVIDIA_STUB.replace("__DATA__", str(self.data)))
        self.nvidia.chmod(0o755)

        self.marker_bin = self.tmp / "fake-llama-server"
        self.marker_bin.write_text(f"#!/bin/sh\necho {MARKER}\n")
        self.marker_bin.chmod(0o755)

        # Same marker, plus the argument list: lets tests inspect the real launch args.
        self.args_bin = self.tmp / "fake-llama-args"
        self.args_bin.write_text(f"#!/bin/sh\necho {MARKER}\nprintf '%s ' \"$@\"\necho\n")
        self.args_bin.chmod(0o755)

        self.children = []
        self.reapers = []
        self.addCleanup(self._kill_children)

    def _kill_children(self):
        for p in self.children:
            if p.poll() is None:
                p.terminate()
        for t in self.reapers:
            t.join(timeout=5)
        for p in self.children:
            if p.poll() is None:
                p.kill()
        # Holders block reading their stdin pipe: close it only now that they are
        # gone, so the fd is not leaked (closing earlier would send EOF, ending
        # a holder mid-test).
        for p in self.children:
            if p.stdin is not None:
                p.stdin.close()

    def spawn(self, args, **kwargs):
        # DEVNULL: a leaked/orphaned child must never hold the runner's stdout
        # pipe open, or piped test output cannot reach EOF.
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
        p = subprocess.Popen(args, **kwargs)
        self.children.append(p)
        # Reap promptly: a zombie still answers kill -0, which would keep the
        # stub nvidia-smi reporting the holder after it has exited.
        t = threading.Thread(target=p.wait, daemon=True)
        t.start()
        self.reapers.append(t)
        return p

    def holder_proc(self, sleep_for=100, name="llama-server", directory=None):
        # A real ELF named `name` (a copy of bash): /proc/<pid>/exe resolves to
        # it, so llm-serve identifies this holder by its executable, exactly as
        # it does for a genuine server. It blocks reading a line from a pipe that
        # stays open, so it lives until terminated. directory may contain spaces.
        directory = self.stub if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        shutil.copy(self.holder_shell, path)
        return self.spawn([str(path), "-c", "read -r _"], stdin=subprocess.PIPE)

    def holder(self, sleep_for=100):
        return self.holder_proc(sleep_for).pid

    def non_ours_process(self):
        return self.spawn(["sleep", "30"]).pid

    def write_holders(self, *rows):
        self.data.write_text("".join(f"{pid}, {name}, {mem}\n" for pid, name, mem in rows))

    def install_nvidia_stub(self, text):
        self.nvidia.write_text(text)
        self.nvidia.chmod(0o755)

    def serve_env(self, port, **env):
        base = {
            "PATH": f"{self.stub}:{os.environ['PATH']}",
            "HOME": str(self.tmp / "home"),
            "MODELS": str(self.models),
            "LOG_DIR": str(self.logs),
            "LLAMA_BIN": str(self.marker_bin),
            "OPENCODE_SYNC": "0",
            "PORT": str(port),
        }
        base.update(env)
        return base

    def run_serve(self, port, profile="qwen-llama", **env):
        return subprocess.run(
            [str(SCRIPT), profile],
            capture_output=True, text=True, timeout=30, env=self.serve_env(port, **env))

    def wait_for_line(self, proc, needle, timeout=15):
        """Read proc stdout line by line until needle appears; fail otherwise."""
        pipe = proc.stdout
        if pipe is None:
            self.fail("stdout pipe missing")
        seen = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and needle not in seen:
            if select.select([pipe], [], [], 0.1)[0]:
                line = pipe.readline()
                if not line:
                    break
                seen += line
        self.assertIn(needle, seen, "expected output never appeared")
        return seen

    def test_no_holders_launches(self):
        self.write_holders()
        out = self.run_serve(free_port())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_holder_exits_then_launches(self):
        holder = self.holder_proc(100)   # long-lived: the test kills it, not a timer
        self.write_holders((holder.pid, "/tmp/llama-server", 90000))
        port = free_port()
        proc = subprocess.Popen([str(SCRIPT), "qwen-llama"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=self.serve_env(port, GPU_WAIT_S="30"))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        seen = self.wait_for_line(proc, WAIT_MSG)
        holder.terminate()
        out, err = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 0, seen + out + err)
        self.assertIn(WAIT_MSG, seen)
        self.assertIn(MARKER, out)

    def test_holder_stays_refuses(self):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        start = time.monotonic()
        out = self.run_serve(free_port(), GPU_WAIT_S="2")
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 1.9, f"waited only {elapsed:.2f}s")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn(str(pid), out.stderr)
        self.assertIn("Stop pid", out.stderr)
        self.assertIn("llama-server", out.stderr)
        self.assertIn("ALLOW_SHARED_GPU=1", out.stderr)
        self.assertNotIn("Stop it with: llm-serve --stop,", out.stderr)
        self.assertNotIn(MARKER, out.stdout)
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_invalid_gpu_wait_s_refuses(self):
        self.write_holders()
        out = self.run_serve(free_port(), GPU_WAIT_S="abc")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("GPU_WAIT_S", out.stderr)
        self.assertNotIn(MARKER, out.stdout)
        self.assertEqual(list(self.logs.iterdir()) if self.logs.exists() else [], [])

    def test_comma_separated_process_name_keeps_ours(self):
        pid = self.holder(100)
        other = self.non_ours_process()
        self.write_holders(
            (pid, "/tmp/llama-server", 90000),
            (other, "/opt/google/chrome/chrome --type=gpu-process --x=a,b,c", 500),
        )
        out = self.run_serve(free_port(), GPU_WAIT_S="0")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("90000 MiB", out.stdout)
        self.assertIn(str(pid), out.stdout)
        self.assertNotIn("chrome", out.stdout)

    def test_process_name_with_spaces_and_args(self):
        # A holder whose /proc exe is a stub binary named llama-server, launched
        # through a path that contains spaces. Identifying by the real executable
        # is immune to the spaces and never misled by the whole command line
        # nvidia-smi reports.
        spaced_dir = self.tmp / "my engines" / "llama.cpp build"
        spaced = self.holder_proc(100, directory=spaced_dir)
        with_args = self.holder_proc(100)
        self.write_holders(
            (spaced.pid, f"{spaced_dir / 'llama-server'} --host 127.0.0.1 --metrics", 90000),
            (with_args.pid, "/opt/llama.cpp/build/bin/llama-server --host 127.0.0.1 --metrics", 80000),
        )
        out = self.run_serve(free_port(), GPU_WAIT_S="0")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn(WAIT_MSG, out.stdout)
        self.assertIn(str(spaced.pid), out.stderr)
        self.assertIn(str(with_args.pid), out.stderr)
        self.assertIn("llama-server", out.stderr)
        self.assertNotIn(MARKER, out.stdout)

    def test_deleted_exe_link_still_counts(self):
        # A binary rebuilt while its process runs: the kernel reports the exe
        # link as "/path/llama-server (deleted)", so the bare basename check
        # would miss it. A fake PROC_ROOT whose <pid>/exe symlink dangles onto
        # such a target must still count the pid as a llama-server holder.
        pid = self.non_ours_process()
        fake_proc = self.tmp / "fake-proc"
        (fake_proc / str(pid)).mkdir(parents=True)
        os.symlink("/x/build/bin/llama-server (deleted)", fake_proc / str(pid) / "exe")
        self.write_holders((pid, "/x/build/bin/llama-server (deleted) --host 127.0.0.1", 90000))
        out = self.run_serve(free_port(), GPU_WAIT_S="0", PROC_ROOT=str(fake_proc))
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn(WAIT_MSG, out.stdout)
        self.assertIn(str(pid), out.stdout)
        self.assertIn("llama-server", out.stdout)
        self.assertNotIn(MARKER, out.stdout)

    def test_leading_zero_gpu_wait_s_honoured(self):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        start = time.monotonic()
        out = self.run_serve(free_port(), GPU_WAIT_S="02")
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 1.9, f"waited only {elapsed:.2f}s")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        for bad in ("arithmetic", "syntax error", "value too great"):
            self.assertNotIn(bad, out.stderr)

    def test_too_long_gpu_wait_s_refuses(self):
        self.write_holders()
        out = self.run_serve(free_port(), GPU_WAIT_S="1234567")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("GPU_WAIT_S", out.stderr)
        self.assertNotIn(MARKER, out.stdout)

    def test_nvidia_smi_failure_mid_wait_keeps_waiting(self):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        counter = self.stub / "count"
        self.install_nvidia_stub(FAIL_MID_NVIDIA_STUB
                                 .replace("__DATA__", str(self.data))
                                 .replace("__COUNTER__", str(counter)))
        out = self.run_serve(free_port(), GPU_WAIT_S="2")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertNotIn(MARKER, out.stdout)
        self.assertIn(str(pid), out.stderr)

    def test_poll_timeout_capped_by_deadline(self):
        # A poll that blocks (nvidia-smi hangs) must not overrun GPU_WAIT_S by a
        # whole poll: the poll timeout is capped at the seconds left. The stub
        # hangs 30 s on the second call (the first wait poll); with GPU_WAIT_S=2
        # the refusal must arrive within about 5 s, not ~10-12 s.
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        counter = self.stub / "pollcount"
        self.install_nvidia_stub(SLOW_SECOND_NVIDIA_STUB
                                 .replace("__DATA__", str(self.data))
                                 .replace("__COUNTER__", str(counter)))
        start = time.monotonic()
        out = self.run_serve(free_port(), GPU_WAIT_S="2")
        elapsed = time.monotonic() - start
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLess(elapsed, 5, f"poll not capped by deadline: {elapsed:.1f}s")
        self.assertIn(str(pid), out.stderr)
        self.assertNotIn(MARKER, out.stdout)

    def test_name_only_maya_holder_waits(self):
        # The pid is a real sleep (its /proc exe resolves to sleep, not maya-ds4),
        # so point PROC_ROOT at a nonexistent dir to make the exe read fail and
        # exercise the name fallback, which the fake name then matches.
        pid = self.non_ours_process()
        self.write_holders((pid, "/x/bin/maya-ds4", 91000))
        out = self.run_serve(free_port(), GPU_WAIT_S="0", PROC_ROOT=str(self.no_proc))
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn(WAIT_MSG, out.stdout)
        self.assertIn("maya-ds4", out.stdout)

    def test_name_only_basalt_helper_ignored(self):
        # Same fallback path (exe unreadable): the name fallback must match the
        # executable part exactly, so basalt-helper is not basalt.
        pid = self.non_ours_process()
        self.write_holders((pid, "/x/bin/basalt-helper", 500))
        out = self.run_serve(free_port(), PROC_ROOT=str(self.no_proc))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_arg_mentioning_our_name_not_counted(self):
        # A non-ours process whose nvidia-smi command line merely mentions
        # /tmp/llama-server in an argument must not be counted. A real sleep's
        # /proc exe resolves to sleep, so the executable check rejects it.
        pid = self.non_ours_process()
        self.write_holders(
            (pid, "python3 /opt/train/run.py --save /tmp/llama-server --epochs 10", 500))
        out = self.run_serve(free_port())
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_arg_mentioning_our_name_not_counted_by_name_fallback(self):
        # Same command line, but with the exe unreadable (PROC_ROOT fake) the
        # name fallback is used; the reference is past the first " -", i.e. in
        # the arguments, so the executable part (run.py) still does not match.
        pid = self.non_ours_process()
        self.write_holders(
            (pid, "python3 /opt/train/run.py --save /tmp/llama-server --epochs 10", 500))
        out = self.run_serve(free_port(), PROC_ROOT=str(self.no_proc))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_port_taken_during_wait_refuses(self):
        holder = self.holder_proc(100)   # long-lived: stays until we kill it
        self.write_holders((holder.pid, "/tmp/llama-server", 90000))
        port = free_port()
        proc = subprocess.Popen([str(SCRIPT), "qwen-llama"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=self.serve_env(port, GPU_WAIT_S="30"))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        seen = self.wait_for_line(proc, WAIT_MSG)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        try:
            holder.terminate()           # GPU frees while the port is taken
            out, err = proc.communicate(timeout=15)
        finally:
            sock.close()
        self.assertEqual(proc.returncode, 1, seen + out + err)
        self.assertIn("Something is already listening", err)
        self.assertNotIn(MARKER, seen + out + err)

    def test_busy_port_fails_fast_before_gpu_wait(self):
        # A GPU holder is present AND the port is taken: report the busy port
        # immediately instead of waiting GPU_WAIT_S for the GPU.
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        port = free_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        self.addCleanup(sock.close)
        start = time.monotonic()
        out = self.run_serve(port, GPU_WAIT_S="30")
        elapsed = time.monotonic() - start
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertLess(elapsed, 3, f"took {elapsed:.1f}s instead of failing fast")
        self.assertIn("Something is already listening", out.stderr)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_unknown_profile_fails_fast_before_gpu_wait(self):
        # A typo (profile or mistyped flag) must fail instantly, not after a
        # GPU_WAIT_S-long GPU wait.
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        for bad, msg in (("qwn", "Unknown profile or flag: qwn"),
                         ("--stauts", "Unknown profile or flag: --stauts")):
            start = time.monotonic()
            out = self.run_serve(free_port(), profile=bad, GPU_WAIT_S="30")
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 3, f"{bad}: {elapsed:.1f}s instead of failing fast")
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            self.assertIn(msg, out.stderr)
            self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def interrupt_during_wait(self, sig, profile="qwen-llama", extra_env=None):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        proc = subprocess.Popen([str(SCRIPT), profile],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=self.serve_env(free_port(), GPU_WAIT_S="30",
                                                              **(extra_env or {})))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        seen = ""
        try:
            seen = self.wait_for_line(proc, WAIT_MSG)
            proc.send_signal(sig)
            out, err = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertTrue(proc.returncode < 0 or proc.returncode == 130,
                        f"llm-serve exited with {proc.returncode} instead of dying on {sig.name}")
        self.assertNotIn(MARKER, seen + out + err)
        return proc, seen + out + err

    def test_sigint_during_wait_dies(self):
        self.interrupt_during_wait(signal.SIGINT)

    def test_sigterm_during_wait_dies(self):
        self.interrupt_during_wait(signal.SIGTERM)

    def interrupt_during_config_generation(self, sig):
        # The GPU wait runs before profile setup, so interrupting there never
        # sees a CONFIG_TMP. The qwen profile instead generates its config with
        # jq: a stub jq that leaves a marker, sleeps, then execs the real jq
        # holds llm-serve right after mktemp put CONFIG_TMP in TMPDIR. bash
        # defers the signal trap until jq exits, so the cleanup runs while the
        # temp file still exists and must remove it before llm-serve dies.
        real_jq = shutil.which("jq")
        if real_jq is None:
            self.skipTest("jq not found on this machine")
        tmpdir = self.tmp / "tmpdir"
        tmpdir.mkdir()
        started = self.tmp / "jq-stub-started"
        jq_stub = self.stub / "jq"
        jq_stub.write_text(
            "#!/bin/sh\n"
            f"printf x > '{started}'\n"
            "sleep 5\n"
            f"exec '{real_jq}' \"$@\"\n"
        )
        jq_stub.chmod(0o755)
        proc = subprocess.Popen([str(SCRIPT), "qwen"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True,
                                env=self.serve_env(free_port(), TMPDIR=str(tmpdir)))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not started.exists():
            time.sleep(0.05)
        self.assertTrue(started.exists(), "qwen's jq was never called: no config was generated")
        proc.send_signal(sig)
        try:
            out, err = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertLess(proc.returncode, 0,
                        f"llm-serve exited with {proc.returncode} instead of dying on {sig.name}")
        self.assertEqual(list(tmpdir.iterdir()), [], "CONFIG_TMP survived the interrupt")
        self.assertNotIn(MARKER, out + err)
        self.assertFalse((self.logs / "basalt-qwen.json").exists(),
                         "the live config was written despite the interrupt")

    def test_sigint_after_config_created_removes_tmp(self):
        self.interrupt_during_config_generation(signal.SIGINT)

    def test_sigterm_after_config_created_removes_tmp(self):
        self.interrupt_during_config_generation(signal.SIGTERM)

    def test_inherited_config_tmp_is_untouched(self):
        # An inherited CONFIG_TMP env var must not become the EXIT trap's victim.
        victim = self.tmp / "victim"
        victim.mkdir()
        victim_file = victim / "secret.json"
        victim_file.write_text("hello")
        port = free_port()
        for args in (["--help"], ["--status", "--port", str(port)], ["no-such-profile"]):
            out = subprocess.run([str(SCRIPT), *args], capture_output=True, text=True,
                                 timeout=30, env=self.serve_env(port, CONFIG_TMP=str(victim_file)))
            self.assertTrue(victim_file.exists(),
                            f"{' '.join(args)} deleted an inherited CONFIG_TMP (rc={out.returncode})")
        self.assertEqual(victim_file.read_text(), "hello")

    def test_help_names_basalt_056_builds(self):
        port = free_port()
        out = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True,
                             timeout=30, env=self.serve_env(port))
        text = out.stdout + out.stderr
        self.assertEqual(out.returncode, 0, text)
        self.assertIn("~/dev/engines/basalt-056", text)
        self.assertIn("~/dev/engines/basalt-ds41-056", text)
        self.assertNotIn("default: ~/dev/basalt-src", text)
        self.assertNotIn("basalt-ds41-serve", text)

    def test_dry_run_qwen_defaults_to_basalt_056(self):
        out = self.run_serve(free_port(), profile="qwen", DRY_RUN="1")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(str(self.tmp / "home" / "dev" / "engines" / "basalt-056" / "bin" / "basalt"),
                      out.stdout)
        self.assertNotIn("basalt-src", out.stdout)

    def test_stop_does_not_kill_maya_ds4_lookalike(self):
        # Make global pgrep find nothing, so --stop can never signal a real
        # llama-server that happens to be running on the developer machine.
        pgrep = self.stub / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 1\n")
        pgrep.chmod(0o755)

        directory = self.tmp / "maya-ds4"
        directory.mkdir()
        port = free_port()
        p = subprocess.Popen([sys.executable, "-m", "http.server", str(port),
                              "--bind", "127.0.0.1", "--directory", str(directory)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: p.poll() is None and p.kill())
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("http.server did not start")
            out = subprocess.run([str(SCRIPT), "--stop", str(port)],
                                 capture_output=True, text=True, timeout=30,
                                 env=self.serve_env(port))
            self.assertIsNone(p.poll(), "non-ours process was killed")
            self.assertIn("not a server we manage", out.stdout + out.stderr)
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    def test_allow_shared_gpu_launches(self):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        out = self.run_serve(free_port(), ALLOW_SHARED_GPU="1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_non_ours_gpu_process_ignored(self):
        pid = self.non_ours_process()
        self.write_holders((pid, "/usr/bin/Xwayland", 500))
        out = self.run_serve(free_port())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_nvidia_smi_unavailable_launches(self):
        # no data file: the stub nvidia-smi exits 1, like a missing/failing one
        out = self.run_serve(free_port())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def test_missing_timeout_still_detects_holder(self):
        # A box without coreutils' timeout: nvidia-smi must still be queried
        # (unguarded) and the holder still found. PATH is the stub dir plus a
        # dir of symlinks to just the commands the launch path needs.
        needed = ("bash date dirname grep mkdir readlink sleep ss")
        bindir = self.tmp / "no-timeout-bin"
        bindir.mkdir()
        for cmd in needed.split():
            src = shutil.which(cmd)
            if src is None:
                self.skipTest(f"{cmd} not found on this machine")
            os.symlink(src, bindir / cmd)
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        port = free_port()
        env = self.serve_env(port, GPU_WAIT_S="0")
        env["PATH"] = f"{self.stub}:{bindir}"
        for d in env["PATH"].split(":"):
            self.assertFalse(os.access(os.path.join(d, "timeout"), os.X_OK),
                             "timeout must not be reachable on the test PATH")
        out = subprocess.run([str(SCRIPT), "qwen-llama"], capture_output=True,
                             text=True, timeout=30, env=env)
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn(WAIT_MSG, out.stdout)
        self.assertIn(str(pid), out.stderr)
        self.assertIn("llama-server", out.stderr)
        self.assertIn("ALLOW_SHARED_GPU=1", out.stderr)
        self.assertNotIn("could not query nvidia-smi", out.stderr)
        self.assertNotIn("timeout", out.stderr)
        self.assertNotIn(MARKER, out.stdout)

    def test_nvidia_smi_ignoring_term_fails_open(self):
        # The stub ignores SIGTERM; only timeout's --kill-after ends it.
        self.install_nvidia_stub(TERM_NVIDIA_STUB)
        start = time.monotonic()
        out = self.run_serve(free_port())
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15, f"hung for {elapsed:.1f}s")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(MARKER, out.stdout)
        self.assertIn("could not query nvidia-smi", out.stderr)

    def test_moe_cache_sized_after_the_wait(self):
        # deepseek-llama must size its MoE cache from the VRAM freed by the
        # wait, not from the reading taken while the holder is still up.
        holder = self.holder_proc(100)
        self.write_holders((holder.pid, "/tmp/llama-server", 90000))
        self.install_nvidia_stub(FLEX_NVIDIA_STUB
                                 .replace("__HOLDER__", str(holder.pid))
                                 .replace("__DATA__", str(self.data)))
        port = free_port()
        proc = subprocess.Popen([str(SCRIPT), "deepseek-llama"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True,
                                env=self.serve_env(port, GPU_WAIT_S="30",
                                                   LLAMA_BIN=str(self.args_bin)))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        seen = self.wait_for_line(proc, WAIT_MSG)
        holder.terminate()
        out, err = proc.communicate(timeout=30)
        allout = seen + out + err
        self.assertEqual(proc.returncode, 0, allout)
        want = 90000 - 3072 - 23000   # free-after - reserve - non-cache
        self.assertIn(f"--moe-cache-mib {want}", out)
        self.assertIn("free 90000", out)
        self.assertNotIn("free 2000", out)

    def test_dry_run_skips_check(self):
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        out = self.run_serve(free_port(), DRY_RUN="1", GPU_WAIT_S="1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(str(self.marker_bin), out.stdout)
        self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)
        self.assertNotIn(MARKER, out.stdout)

    def test_dry_run_accepts_every_profile(self):
        # The early profile check must accept exactly the names the profile
        # case accepts after the alias mapping, or a valid name dies as
        # "Unknown profile" before DRY_RUN ever prints a command.
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        ds_dir = self.tmp / "basalt-ds-dir"
        (ds_dir / "bin").mkdir(parents=True)
        ds_basalt = ds_dir / "bin" / "basalt"
        ds_basalt.write_text("#!/bin/sh\nexit 0\n")
        ds_basalt.chmod(0o755)
        ds_python = self.tmp / "basalt-ds-python"
        ds_python.write_text("#!/bin/sh\nexit 0\n")
        ds_python.chmod(0o755)
        maya_dir = self.tmp / "maya-dir"
        maya_dir.mkdir()
        (maya_dir / "maya-ds4.json").write_text(
            '{"args": ["--max-context", "131072", "--reserve-mib", "4096",'
            ' "--ablate", "refusal.gguf"], "ablate": {"refusal": 1},'
            ' "sampling": {"ablate": 2}}\n'
        )
        stub_env = {
            "DRY_RUN": "1",
            "BASALT_DS_DIR": str(ds_dir),
            "BASALT_DS_PYTHON": str(ds_python),
            "MAYA_DIR": str(maya_dir),
            "MOE_CACHE_MIB": "65536",
        }
        for profile in ("qwen", "deepseek", "deepseek-maya", "qwen-llama",
                        "deepseek-llama", "qwen-basalt", "deepseek-basalt"):
            out = self.run_serve(free_port(), profile=profile, **stub_env)
            self.assertEqual(out.returncode, 0,
                             f"{profile}: {out.stdout}{out.stderr}")
            self.assertNotIn("Unknown profile", out.stdout + out.stderr)
        # unknown names still fail fast: a GPU holder is up and GPU_WAIT_S=30,
        # so a check that ran after the wait would take 30 s to answer.
        pid = self.holder(100)
        self.write_holders((pid, "/tmp/llama-server", 90000))
        for bad in ("deepseek-mayas", "qwen-maya", "deepseek-basal"):
            start = time.monotonic()
            out = self.run_serve(free_port(), profile=bad, GPU_WAIT_S="30")
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 3, f"{bad}: {elapsed:.1f}s instead of failing fast")
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            self.assertIn(f"Unknown profile or flag: {bad}", out.stderr)
            self.assertNotIn(WAIT_MSG, out.stdout + out.stderr)

    def dry_run_config(self, profile, **env):
        # DRY_RUN prints the launch command, then "# <config path>" and the config.
        out = self.run_serve(free_port(), profile=profile, DRY_RUN="1", **env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        _, sep, config = out.stdout.partition("\n# ")
        self.assertTrue(sep, out.stdout)
        return json.loads(config.split("\n", 1)[1])

    def test_qwen_vision(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        cfg = self.dry_run_config("qwen")
        self.assertEqual(cfg["engine"]["vision"], "dynamic")
        self.assertNotIn("vision", cfg["server"])
        for value, want in (("cpu", "cpu"), ("off", "off"), ("1", 1), ("dynamic", "dynamic")):
            engine = self.dry_run_config("qwen", VISION=value)["engine"]
            self.assertEqual(engine["vision"], want, value)
            self.assertIs(type(engine["vision"]), type(want), value)

    def test_invalid_vision_refuses_before_starting(self):
        basalt_dir = self.tmp / "basalt-dir"
        (basalt_dir / "bin").mkdir(parents=True)
        basalt = basalt_dir / "bin" / "basalt"
        basalt.write_text(f"#!/bin/sh\necho {MARKER}\n")
        basalt.chmod(0o755)
        self.write_holders()
        for bad in ("gpu", "-1", "1.5", "GPU0"):
            for dry in ("0", "1"):
                out = self.run_serve(free_port(), profile="qwen", VISION=bad,
                                     BASALT_DIR=str(basalt_dir), DRY_RUN=dry)
                self.assertEqual(out.returncode, 1, f"{bad!r}: {out.stdout}{out.stderr}")
                self.assertIn("VISION must be", out.stderr)
                self.assertNotIn(MARKER, out.stdout)
                self.assertNotIn(str(basalt), out.stdout)
        self.assertFalse(self.logs.exists())

    def test_qwen_conversation_cache(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        engine = self.dry_run_config("qwen")["engine"]
        for key, want in (("conversation-cache-mib", 16384),
                          ("conversation-cache-slots", 4),
                          ("conversation-cache-min-tokens", 32768),
                          ("conversation-cache-disk-mib", 65536)):
            self.assertEqual(engine[key], want, key)
            self.assertIs(type(engine[key]), int, key)
        # The qwen-basalt alias is qwen: same defaults.
        engine = self.dry_run_config("qwen-basalt")["engine"]
        self.assertEqual(engine["conversation-cache-mib"], 16384)
        self.assertEqual(engine["conversation-cache-slots"], 4)
        # RAM tier off: its key goes, but min-tokens stays while the disk tier is on.
        engine = self.dry_run_config("qwen", CONV_CACHE_MIB="0", CONV_CACHE_DISK_MIB="65536",
                                     CONV_CACHE_MIN_TOKENS="65536")["engine"]
        self.assertNotIn("conversation-cache-mib", engine)
        self.assertNotIn("conversation-cache-slots", engine)
        self.assertEqual(engine["conversation-cache-disk-mib"], 65536)
        self.assertEqual(engine["conversation-cache-min-tokens"], 65536)
        # 0 slots is accepted once the RAM tier is off: the launcher skips that rule.
        engine = self.dry_run_config("qwen", CONV_CACHE_MIB="0", CONV_CACHE_SLOTS="0")["engine"]
        self.assertNotIn("conversation-cache-mib", engine)
        self.assertNotIn("conversation-cache-slots", engine)
        engine = self.dry_run_config("qwen", CONV_CACHE_DISK_MIB="0")["engine"]
        self.assertNotIn("conversation-cache-disk-mib", engine)
        self.assertEqual(engine["conversation-cache-mib"], 16384)
        self.assertEqual(engine["conversation-cache-min-tokens"], 32768)
        # Both tiers off: no key at all, whatever min-tokens says.
        engine = self.dry_run_config("qwen", CONV_CACHE_MIB="0", CONV_CACHE_DISK_MIB="0",
                                     CONV_CACHE_MIN_TOKENS="65536")["engine"]
        for key in ("conversation-cache-mib", "conversation-cache-slots",
                    "conversation-cache-min-tokens", "conversation-cache-disk-mib"):
            self.assertNotIn(key, engine)
        engine = self.dry_run_config("qwen", CONV_CACHE_MIB="2048", CONV_CACHE_SLOTS="8",
                                     CONV_CACHE_MIN_TOKENS="4096",
                                     CONV_CACHE_DISK_MIB="131072")["engine"]
        for key, want in (("conversation-cache-mib", 2048),
                          ("conversation-cache-slots", 8),
                          ("conversation-cache-min-tokens", 4096),
                          ("conversation-cache-disk-mib", 131072)):
            self.assertEqual(engine[key], want, key)
            self.assertIs(type(engine[key]), int, key)

    def test_invalid_conversation_cache_refuses_before_starting(self):
        basalt_dir = self.tmp / "basalt-dir"
        (basalt_dir / "bin").mkdir(parents=True)
        basalt = basalt_dir / "bin" / "basalt"
        basalt.write_text(f"#!/bin/sh\necho {MARKER}\n")
        basalt.chmod(0o755)
        self.write_holders()
        for name in ("CONV_CACHE_MIB", "CONV_CACHE_MIN_TOKENS", "CONV_CACHE_DISK_MIB",
                     "CONV_CACHE_SLOTS"):
            # "" stays refused (an explicitly set value wins over the profile
            # default), "012" has a leading zero and "1000000000" is more than
            # 9 digits.
            for bad in ("-1", "1.5", "abc", "", "012", "1000000000"):
                for dry in ("0", "1"):
                    out = self.run_serve(free_port(), profile="qwen", BASALT_DIR=str(basalt_dir),
                                         DRY_RUN=dry, **{name: bad})
                    self.assertEqual(out.returncode, 1, f"{name}={bad!r}: {out.stdout}{out.stderr}")
                    self.assertIn(f"{name} must be a non-negative integer of at most 9 digits (got: {bad})", out.stderr)
                    self.assertNotIn(MARKER, out.stdout)
                    self.assertNotIn(str(basalt), out.stdout)
        # 0 slots is refused with its own message: CONV_CACHE_MIB=0 is how the
        # RAM tier gets turned off, not a slot count of zero.
        for dry in ("0", "1"):
            out = self.run_serve(free_port(), profile="qwen", BASALT_DIR=str(basalt_dir),
                                 DRY_RUN=dry, CONV_CACHE_SLOTS="0")
            self.assertEqual(out.returncode, 1, f"dry={dry}: {out.stdout}{out.stderr}")
            self.assertIn("CONV_CACHE_SLOTS must be at least 1 (got: 0)", out.stderr)
            self.assertNotIn(MARKER, out.stdout)
            self.assertNotIn(str(basalt), out.stdout)
        self.assertFalse(self.logs.exists())

    def test_deepseek_basalt_conversation_cache(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        ds_dir = self.tmp / "basalt-ds-dir"
        (ds_dir / "bin").mkdir(parents=True)
        ds_basalt = ds_dir / "bin" / "basalt"
        ds_basalt.write_text(f"#!/bin/sh\necho {MARKER}\n")
        ds_basalt.chmod(0o755)
        ds_python = self.tmp / "basalt-ds-python"
        ds_python.write_text("#!/bin/sh\nexit 0\n")
        ds_python.chmod(0o755)
        # DeepSeek prompts cost about 30x less KV than Qwen's, so its cache
        # defaults are smaller: 2048 / 8192 / 16384.
        env = {"BASALT_DS_DIR": str(ds_dir), "BASALT_DS_PYTHON": str(ds_python)}
        engine = self.dry_run_config("deepseek", **env)["engine"]
        for key, want in (("conversation-cache-mib", 2048),
                          ("conversation-cache-slots", 16),
                          ("conversation-cache-min-tokens", 8192),
                          ("conversation-cache-disk-mib", 16384)):
            self.assertEqual(engine[key], want, key)
            self.assertIs(type(engine[key]), int, key)
        engine = self.dry_run_config("deepseek", CONV_CACHE_MIB="1024", CONV_CACHE_SLOTS="32",
                                     CONV_CACHE_MIN_TOKENS="512",
                                     CONV_CACHE_DISK_MIB="2048", **env)["engine"]
        for key, want in (("conversation-cache-mib", 1024),
                          ("conversation-cache-slots", 32),
                          ("conversation-cache-min-tokens", 512),
                          ("conversation-cache-disk-mib", 2048)):
            self.assertEqual(engine[key], want, key)
            self.assertIs(type(engine[key]), int, key)
        # Only one knob set: it wins, the other three still take deepseek defaults.
        engine = self.dry_run_config("deepseek", CONV_CACHE_MIB="4096", **env)["engine"]
        for key, want in (("conversation-cache-mib", 4096),
                          ("conversation-cache-slots", 16),
                          ("conversation-cache-min-tokens", 8192),
                          ("conversation-cache-disk-mib", 16384)):
            self.assertEqual(engine[key], want, key)
        # The deepseek-basalt alias is deepseek: same defaults.
        engine = self.dry_run_config("deepseek-basalt", **env)["engine"]
        self.assertEqual(engine["conversation-cache-mib"], 2048)
        self.assertEqual(engine["conversation-cache-slots"], 16)
        # 0 slots is accepted once the RAM tier is off: the launcher skips that rule.
        engine = self.dry_run_config("deepseek", CONV_CACHE_MIB="0", CONV_CACHE_SLOTS="0",
                                     **env)["engine"]
        self.assertNotIn("conversation-cache-mib", engine)
        self.assertNotIn("conversation-cache-slots", engine)
        # RAM tier off: its key goes, min-tokens stays while the disk tier is on.
        engine = self.dry_run_config("deepseek", CONV_CACHE_MIB="0", **env)["engine"]
        self.assertNotIn("conversation-cache-mib", engine)
        self.assertNotIn("conversation-cache-slots", engine)
        self.assertEqual(engine["conversation-cache-min-tokens"], 8192)
        self.assertEqual(engine["conversation-cache-disk-mib"], 16384)
        # Both tiers off: no key at all, whatever min-tokens says.
        engine = self.dry_run_config("deepseek", CONV_CACHE_MIB="0", CONV_CACHE_DISK_MIB="0",
                                     CONV_CACHE_MIN_TOKENS="65536", **env)["engine"]
        for key in ("conversation-cache-mib", "conversation-cache-slots",
                    "conversation-cache-min-tokens", "conversation-cache-disk-mib"):
            self.assertNotIn(key, engine)
        # Deepseek validates the knobs too, before Basalt starts.
        self.write_holders()
        for name in ("CONV_CACHE_MIB", "CONV_CACHE_MIN_TOKENS", "CONV_CACHE_DISK_MIB",
                     "CONV_CACHE_SLOTS"):
            for bad in ("-1", "1.5", "abc", "", "012", "1000000000"):
                for dry in ("0", "1"):
                    out = self.run_serve(free_port(), profile="deepseek",
                                         BASALT_DS_DIR=str(ds_dir),
                                         BASALT_DS_PYTHON=str(ds_python),
                                         DRY_RUN=dry, **{name: bad})
                    self.assertEqual(out.returncode, 1, f"{name}={bad!r}: {out.stdout}{out.stderr}")
                    self.assertIn(f"{name} must be a non-negative integer of at most 9 digits "
                                  f"(got: {bad})", out.stderr)
                    self.assertNotIn(MARKER, out.stdout)
                    self.assertNotIn(str(ds_basalt), out.stdout)
        # And 0 slots refuses while the RAM tier is on, with its own message.
        for dry in ("0", "1"):
            out = self.run_serve(free_port(), profile="deepseek", CONV_CACHE_SLOTS="0",
                                 BASALT_DS_DIR=str(ds_dir), BASALT_DS_PYTHON=str(ds_python),
                                 DRY_RUN=dry)
            self.assertEqual(out.returncode, 1, f"dry={dry}: {out.stdout}{out.stderr}")
            self.assertIn("CONV_CACHE_SLOTS must be at least 1 (got: 0)", out.stderr)
            self.assertNotIn(MARKER, out.stdout)
            self.assertNotIn(str(ds_basalt), out.stdout)
        self.assertFalse(self.logs.exists())

    def test_qwen_llama_ignores_conversation_cache(self):
        out = self.run_serve(free_port(), profile="qwen-llama", DRY_RUN="1",
                             CONV_CACHE_MIB="abc", CONV_CACHE_MIN_TOKENS="abc",
                             CONV_CACHE_DISK_MIB="abc", CONV_CACHE_SLOTS="abc")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNotIn("conversation-cache", out.stdout)

    def test_deepseek_llama_ignores_conversation_cache(self):
        out = self.run_serve(free_port(), profile="deepseek-llama", DRY_RUN="1",
                             MOE_CACHE_MIB="65536",
                             CONV_CACHE_MIB="abc", CONV_CACHE_MIN_TOKENS="abc",
                             CONV_CACHE_DISK_MIB="abc", CONV_CACHE_SLOTS="abc")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNotIn("conversation-cache", out.stdout)

    def test_deepseek_maya_ignores_conversation_cache(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        maya_dir = self.maya_dir_with({"args": ["--max-context", "524288"]})
        out = self.run_serve(free_port(), profile="deepseek-maya", DRY_RUN="1",
                             MAYA_DIR=maya_dir, CONV_CACHE_MIB="abc", CONV_CACHE_SLOTS="abc")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNotIn("conversation-cache", out.stdout)

    def test_deepseek_basalt_vision(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        ds_dir = self.tmp / "basalt-ds-dir"
        (ds_dir / "bin").mkdir(parents=True)
        ds_basalt = ds_dir / "bin" / "basalt"
        ds_basalt.write_text("#!/bin/sh\nexit 0\n")
        ds_basalt.chmod(0o755)
        ds_python = self.tmp / "basalt-ds-python"
        ds_python.write_text("#!/bin/sh\nexit 0\n")
        ds_python.chmod(0o755)
        env = {"BASALT_DS_DIR": str(ds_dir), "BASALT_DS_PYTHON": str(ds_python)}
        cfg = self.dry_run_config("deepseek", **env)
        self.assertEqual(cfg["engine"]["vision"], "dynamic")
        self.assertNotIn("vision", cfg["server"])
        self.assertTrue(cfg["models"][0]["model"].endswith("UD-Q2_K_XL.vision.basalt"),
                        cfg["models"][0]["model"])
        for value, want in (("cpu", "cpu"), ("off", "off"), ("1", 1), ("dynamic", "dynamic")):
            engine = self.dry_run_config("deepseek", VISION=value, **env)["engine"]
            self.assertEqual(engine["vision"], want, value)
            self.assertIs(type(engine["vision"]), type(want), value)

    def test_deepseek_basalt_invalid_vision_refuses_before_starting(self):
        ds_dir = self.tmp / "basalt-ds-dir"
        (ds_dir / "bin").mkdir(parents=True)
        ds_basalt = ds_dir / "bin" / "basalt"
        ds_basalt.write_text(f"#!/bin/sh\necho {MARKER}\n")
        ds_basalt.chmod(0o755)
        self.write_holders()
        for bad in ("gpu", "-1", "1.5", "GPU0", "on"):
            for dry in ("0", "1"):
                out = self.run_serve(free_port(), profile="deepseek", VISION=bad,
                                     BASALT_DS_DIR=str(ds_dir), DRY_RUN=dry)
                self.assertEqual(out.returncode, 1, f"{bad!r}: {out.stdout}{out.stderr}")
                self.assertIn("deepseek: VISION must be", out.stderr)
                self.assertNotIn(MARKER, out.stdout)
                self.assertNotIn(str(ds_basalt), out.stdout)
        self.assertFalse(self.logs.exists())

    def maya_dir_with(self, template):
        maya_dir = self.tmp / "maya-dir"
        maya_dir.mkdir(exist_ok=True)
        (maya_dir / "maya-ds4.json").write_text(json.dumps(template) + "\n")
        return str(maya_dir)

    def test_deepseek_maya_vision(self):
        if shutil.which("jq") is None:
            self.skipTest("jq not found on this machine")
        template = {"args": ["--max-context", "131072", "--vision", "off",
                             "--reserve-mib", "4096", "--ablate", "refusal.gguf",
                             "--ablate-default", "0.5"],
                    "ablate": {"refusal": 1}, "sampling": {"ablate": 2}}
        maya_dir = self.maya_dir_with(template)
        for vision, want in ((None, "on"), ("on", "on"), ("dynamic", "on"), ("off", "off")):
            extra = {"MAYA_DIR": maya_dir, "CTX": "65536"}
            if vision is not None:
                extra["VISION"] = vision
            cfg = self.dry_run_config("deepseek-maya", **extra)
            args = cfg["args"]
            self.assertEqual(args.count("--vision"), 1, args)
            self.assertEqual(args[args.index("--vision") + 1], want, vision)
            # ablation still stripped, ctx/reserve still replaced
            self.assertEqual(args, ["--max-context", "65536", "--vision", want,
                                    "--reserve-mib", "3072"])
            self.assertNotIn("ablate", cfg)
            self.assertNotIn("ablate", cfg["sampling"])
        # a template without --vision gets it appended
        maya_dir = self.maya_dir_with({"args": ["--max-context", "131072"]})
        for vision, want in ((None, "on"), ("off", "off")):
            extra = {"MAYA_DIR": maya_dir}
            if vision is not None:
                extra["VISION"] = vision
            args = self.dry_run_config("deepseek-maya", **extra)["args"]
            self.assertEqual(args, ["--max-context", "524288", "--vision", want])

    def test_deepseek_maya_invalid_vision_refuses_before_starting(self):
        maya_dir = self.maya_dir_with({"args": ["--vision", "off"]})
        self.write_holders()
        for bad in ("cpu", "1", "gpu"):
            for dry in ("0", "1"):
                out = self.run_serve(free_port(), profile="deepseek-maya", VISION=bad,
                                     MAYA_DIR=maya_dir, DRY_RUN=dry)
                self.assertEqual(out.returncode, 1, f"{bad!r}: {out.stdout}{out.stderr}")
                self.assertIn("deepseek-maya: VISION must be on, dynamic or off", out.stderr)
                self.assertNotIn("maya.py", out.stdout)
        self.assertFalse(self.logs.exists())

    def test_qwen_rejects_on(self):
        out = self.run_serve(free_port(), profile="qwen", VISION="on", DRY_RUN="1")
        self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
        self.assertIn("VISION must be off, dynamic, cpu or a GPU number", out.stderr)


if __name__ == "__main__":
    unittest.main()
