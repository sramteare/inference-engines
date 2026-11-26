#!/usr/bin/env python3
"""loadtest.py - measure concurrent generation speed of local LLM servers.

Sessions are fired at one or more OpenAI-compatible endpoints at the same
moment; the speeds are read back from each server's own log (done / live
lines), never from client-side timing.  Standard library only.
"""

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request

WORDS = (
    "the cache keeps recent entries and drops old ones when the limit is reached while each lookup moves an entry to the front so that frequently used keys stay resident and expired keys are removed on read"
).split()

TASKS = (
    "Write a Python module that implements an LRU cache with TTL expiry, with docstrings, type hints and unittest tests.",
    "Write a Python command-line tool that parses a CSV of sales records and prints monthly totals per region, with argparse and tests.",
    "Write a Python asyncio web crawler limited to one domain, with a concurrency limit, retries and tests.",
)

DONE_RE = re.compile(r"\bs(\d+)\s+done\b")
PP_RE = re.compile(r"\bpp=(\d+)/(\d+)c\b")
TG_RE = re.compile(r"\btg=(\d+)\s+([\d.]+)t/s")
ACC_RE = re.compile(r"\bacc=([\d.]+)%")
HIT_RE = re.compile(r"\bhit=([\d.]+)%")
PCIE_RE = re.compile(r"\bpcie=([\d.]+)%")
LIVE_SLOT_RE = re.compile(r"\bs(\d+)\s+tg=(\d+)/(\d+)(?:\s+([\d.]+)t/s)?")
LIVE_TOTAL_RE = re.compile(r"\btotal\s+([\d.]+)t/s")

DONE_WAIT_S = 10.0
POLL_S = 0.25
REQUEST_TIMEOUT_S = 600.0
IDLE_POLL_S = 2.0
IDLE_WAIT_S = 600.0


class Target:
    def __init__(self, spec):
        parts = spec.split(":")
        if len(parts) < 4:
            raise ValueError(f"bad --target {spec!r}, want PORT:LOGFILE:MODEL:COUNT")
        try:
            self.port = int(parts[0])
            self.count = int(parts[-1])
        except ValueError:
            raise ValueError(f"bad --target {spec!r}, PORT and COUNT must be integers")
        if self.count < 1:
            raise ValueError(f"bad --target {spec!r}, COUNT must be >= 1")
        self.model = parts[-2]
        self.log = ":".join(parts[1:-2])


def build_prompt(session_index, ctx_tokens):
    x = session_index * 7919 + 1
    words = []
    for _ in range(ctx_tokens):
        words.append(WORDS[x % len(WORDS)])
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    system = "Reference notes: " + " ".join(words)
    user = TASKS[session_index % 3]
    return system, user


def chat(port, model, system, user, max_tokens, session_id):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.6,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-session-id": session_id,
        },
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8", "replace"))
    usage = body.get("usage") or {}
    choices = body.get("choices") or [{}]
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": choices[0].get("finish_reason"),
        "model": body.get("model"),
    }


def session_worker(session, barrier, prompt, max_tokens, label, results):
    i = session["index"]
    try:
        barrier.wait(timeout=120)
        system, user = prompt
        res = chat(
            session["port"],
            session["model"],
            system,
            user,
            max_tokens,
            f"{label}-s{i}",
        )
        res["session"] = i
        res["target"] = session["tidx"]
        results[i] = res
    except Exception as exc:
        results[i] = {
            "session": i,
            "target": session["tidx"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def tail_lines(path, count=3):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 65536))
            data = fh.read()
    except OSError:
        return []
    lines = [ln for ln in data.decode("utf-8", "replace").split("\n") if ln.strip()]
    return lines[-count:]


def log_is_busy(path):
    tail = tail_lines(path)
    last_live = last_done = None
    for i, ln in enumerate(tail):
        if DONE_RE.search(ln):
            last_done = i
        elif " live | " in ln:
            last_live = i
    return last_live is not None and (last_done is None or last_done < last_live)


def wait_until_idle(target):
    announced = False
    deadline = time.monotonic() + IDLE_WAIT_S
    while log_is_busy(target.log):
        if not announced:
            print(f"waiting for {target.port} to go idle", flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"error: {target.port} never went idle within {IDLE_WAIT_S:g} s"
            )
        time.sleep(IDLE_POLL_S)


def read_new_lines(path, offset):
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    cut = text.rfind("\n")
    if cut < 0:
        return []
    return [ln for ln in text[:cut].split("\n") if ln.strip()]


def wait_for_done(target, offset, timeout=DONE_WAIT_S):
    deadline = time.monotonic() + timeout
    lines = []
    while True:
        lines = read_new_lines(target.log, offset)
        if sum(1 for ln in lines if DONE_RE.search(ln)) >= target.count:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(POLL_S)
    time.sleep(0.25)
    lines = read_new_lines(target.log, offset)
    dones = [ln for ln in lines if DONE_RE.search(ln)]
    return dones, [ln for ln in lines if " live | " in ln]


def parse_done(line):
    m = DONE_RE.search(line)
    if not m:
        return None
    d = {"raw": line.strip(), "slot": "s" + m.group(1)}
    p = PP_RE.search(line)
    d["prompt_tokens"] = int(p.group(1)) if p else None
    d["reused_tokens"] = int(p.group(2)) if p else None
    t = TG_RE.search(line)
    d["generated_tokens"] = int(t.group(1)) if t else None
    d["tps"] = float(t.group(2)) if t else None
    a = ACC_RE.search(line)
    d["acc"] = float(a.group(1)) if a else None
    h = HIT_RE.search(line)
    d["hit"] = float(h.group(1)) if h else None
    c = PCIE_RE.search(line)
    d["pcie"] = float(c.group(1)) if c else None
    return d


def parse_live(line):
    slots = {}
    for m in LIVE_SLOT_RE.finditer(line):
        slots["s" + m.group(1)] = float(m.group(4)) if m.group(4) else None
    tm = LIVE_TOTAL_RE.search(line)
    return {
        "raw": line.strip(),
        "slots": slots,
        "total": float(tm.group(1)) if tm else None,
    }


def median(vals):
    return statistics.median(vals) if vals else None


def fmt(v, suffix="", spec="g"):
    return "n/a" if v is None else f"{v:{spec}}{suffix}"


def fmt_request(port, d):
    return (
        f"  {port} {d['slot']} pp={fmt(d['prompt_tokens'])}/"
        f"{fmt(d['reused_tokens'])} tg={fmt(d['generated_tokens'])} "
        f"{fmt(d['tps'], 't/s')} acc={fmt(d['acc'], '%')} "
        f"hit={fmt(d['hit'], '%')} pcie={fmt(d['pcie'], '%')}"
    )


def run_round(rnum, targets, sessions, prompts, args):
    for t in targets:
        wait_until_idle(t)
    offsets = []
    reasons = []
    for t in targets:
        try:
            offsets.append(os.path.getsize(t.log))
        except FileNotFoundError:
            offsets.append(0)
        except OSError as exc:
            msg = f"round {rnum}: INVALID: cannot read log {t.log}: {exc}"
            print(msg, flush=True)
            reasons.append(msg)
            offsets.append(0)

    results = [None] * len(sessions)
    barrier = threading.Barrier(len(sessions))
    threads = [
        threading.Thread(
            target=session_worker,
            args=(
                s,
                barrier,
                prompts[s["index"]],
                args.max_tokens,
                args.label,
                results,
            ),
        )
        for s in sessions
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    rec = {
        "round": rnum,
        "valid": not reasons,
        "invalid_reasons": list(reasons),
        "targets": [],
        "pool_total": None,
    }

    for ti, t in enumerate(targets):
        dones_raw, lives_raw = wait_for_done(t, offsets[ti])
        dones = [parse_done(ln) for ln in dones_raw]
        dones = [d for d in dones if d]
        lives = [parse_live(ln) for ln in lives_raw]
        tresp = [res for res in results if res and res["target"] == ti]
        trec = {
            "port": t.port,
            "log": t.log,
            "model": t.model,
            "count": t.count,
            "offset_start": offsets[ti],
            "done_lines": [d["raw"] for d in dones],
            "requests": dones,
            "responses": tresp,
            "interference": None,
            "live": None,
        }

        if len(dones) != t.count:
            msg = (
                f"round {rnum}: INTERFERENCE on {t.port}: expected "
                f"{t.count} done lines, saw {len(dones)}"
            )
            print(msg, flush=True)
            trec["interference"] = {"expected": t.count, "saw": len(dones)}
            rec["valid"] = False
            rec["invalid_reasons"].append(msg)
            rec["targets"].append(trec)
            continue

        our_slots = {d["slot"] for d in dones}
        intruder = None
        for lv in lives:
            for sl in lv["slots"]:
                if sl not in our_slots:
                    intruder = sl
                    break
            if intruder is not None:
                break
        if intruder is not None:
            msg = (
                f"round {rnum}: INTERFERENCE on {t.port}: live line "
                f"shows slot {intruder} that is not ours"
            )
            print(msg, flush=True)
            trec["interference"] = {"live_intruder_slot": intruder}
            rec["valid"] = False
            rec["invalid_reasons"].append(msg)

        errs = [
            res
            for res in tresp
            if "error" in res or res.get("completion_tokens") is None
        ]
        for res in errs:
            what = res.get("error", "no usage.completion_tokens in response")
            msg = (
                f"round {rnum}: INVALID on {t.port}: session s{res['session']} "
                f"failed: {what}"
            )
            print(msg, flush=True)
            rec["valid"] = False
            rec["invalid_reasons"].append(msg)
        if not errs:
            tgs = [d["generated_tokens"] for d in dones]
            for res in tresp:
                if res["completion_tokens"] not in tgs:
                    msg = (
                        f"round {rnum}: INVALID on {t.port}: response "
                        f"completion_tokens {res['completion_tokens']} of session "
                        f"s{res['session']} matches no done tg {tgs}"
                    )
                    print(msg, flush=True)
                    rec["valid"] = False
                    rec["invalid_reasons"].append(msg)

        if len(dones) == t.count:
            for d in dones:
                print(fmt_request(t.port, d), flush=True)

        slots = {d["slot"] for d in dones}
        qualifying = [
            lv
            for lv in lives
            if slots
            and all(lv["slots"].get(s) is not None for s in slots)
            and lv["total"] is not None
        ]
        med_total = median([lv["total"] for lv in qualifying])
        trec["live"] = {
            "slots": sorted(slots),
            "all_generating_lines": qualifying,
            "count": len(qualifying),
            "total_median": med_total,
        }
        rec["targets"].append(trec)

    parts = []
    for trec in rec["targets"]:
        if trec["interference"] is not None:
            parts.append(f"{trec['port']}: interference")
        elif trec["live"]["total_median"] is not None:
            parts.append(
                f"{trec['port']} all-generating total median "
                f"{fmt(trec['live']['total_median'], 't/s')} "
                f"({trec['live']['count']} live lines)"
            )
        else:
            parts.append(f"{trec['port']}: no all-generating live line")
    meds = [
        tr["live"]["total_median"]
        for tr in rec["targets"]
        if tr["interference"] is None
        and tr["live"]
        and tr["live"]["total_median"] is not None
    ]
    if rec["valid"] and len(meds) == len(targets):
        rec["pool_total"] = sum(meds)
        parts.append(f"pool total {fmt(rec['pool_total'], 't/s')}")
    else:
        parts.append("pool total n/a")
    print(
        f"round {rnum}: {'valid' if rec['valid'] else 'INVALID'} | "
        + " | ".join(parts),
        flush=True,
    )
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure concurrent generation speed of local LLM servers "
        "from their own logs."
    )
    ap.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="PORT:LOGFILE:MODEL:COUNT",
        help="server endpoint, its log file, model and session count",
    )
    ap.add_argument(
        "--ctx-tokens",
        type=int,
        required=True,
        help="approximate filler tokens in the system prompt",
    )
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--json", dest="json_out", metavar="OUT")
    args = ap.parse_args(argv)

    try:
        targets = [Target(spec) for spec in args.target]
    except ValueError as exc:
        ap.error(str(exc))

    sessions = []
    for ti, t in enumerate(targets):
        for _ in range(t.count):
            sessions.append(
                {"index": len(sessions), "tidx": ti, "port": t.port, "model": t.model}
            )
    prompts = [build_prompt(s["index"], args.ctx_tokens) for s in sessions]
    print(
        "targets: "
        + ", ".join(
            f"{t.port} ({t.model}, {t.count} session(s), log {t.log})" for t in targets
        ),
        flush=True,
    )

    round_recs = [
        run_round(r, targets, sessions, prompts, args)
        for r in range(1, args.rounds + 1)
    ]

    valid = [rc for rc in round_recs if rc["valid"]]
    invalid = [rc for rc in round_recs if not rc["valid"]]
    print("=== FINAL SUMMARY ===")
    print(f"valid rounds: {len(valid)}  invalid rounds: {len(invalid)}")

    summary_targets = {}
    pool_parts = []
    for ti, t in enumerate(targets):
        tps, accs, hits, round_meds = [], [], [], []
        for rc in valid:
            trec = rc["targets"][ti]
            for d in trec["requests"]:
                if d["tps"] is not None:
                    tps.append(d["tps"])
                if d["acc"] is not None:
                    accs.append(d["acc"])
                if d["hit"] is not None:
                    hits.append(d["hit"])
            if trec["live"] and trec["live"]["total_median"] is not None:
                round_meds.append(trec["live"]["total_median"])
        med_tps = median(tps)
        med_total = median(round_meds)
        entry = {
            "port": t.port,
            "model": t.model,
            "sessions": t.count,
            "requests": len(tps),
            "gen_tps_median": med_tps,
            "gen_tps_min": min(tps) if tps else None,
            "gen_tps_max": max(tps) if tps else None,
            "all_generating_total_median": med_total,
            "all_generating_total_rounds": len(round_meds),
            "acc_median": median(accs),
            "hit_median": median(hits) if hits else None,
        }
        summary_targets[str(t.port)] = entry
        pool_parts.append(med_total)
        print(f"port {t.port} ({t.model}, {t.count} session(s)):")
        print(
            f"  gen t/s: median {fmt(med_tps)}, min-max "
            f"{fmt(entry['gen_tps_min'])}-{fmt(entry['gen_tps_max'])} "
            f"({len(tps)} requests over {len(valid)} valid rounds)"
        )
        print(
            f"  all-generating total median: {fmt(med_total, 't/s')} "
            f"({len(round_meds)} rounds)"
        )
        print(f"  acc median: {fmt(entry['acc_median'], '%')}")
        print(f"  hit median: {fmt(entry['hit_median'], '%')}")

    pool_total = (
        sum(pool_parts)
        if all(v is not None for v in pool_parts) and pool_parts
        else None
    )
    if pool_total is None:
        print("pool total: n/a")
    else:
        print(f"pool total: {pool_total:g}t/s")

    out = {
        "label": args.label,
        "ctx_tokens": args.ctx_tokens,
        "max_tokens": args.max_tokens,
        "rounds": args.rounds,
        "targets": [
            {"port": t.port, "log": t.log, "model": t.model, "count": t.count}
            for t in targets
        ],
        "round_records": round_recs,
        "summary": {
            "valid_rounds": len(valid),
            "invalid_rounds": len(invalid),
            "targets": summary_targets,
            "pool_total": pool_total,
        },
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
