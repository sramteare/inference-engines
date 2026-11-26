#!/usr/bin/env python3
"""Alternate two very different conversations on one DeepSeek server and read each request's
decode expert-cache hit rate from the engine log. Usage: hitrate.py PORT ENGINE_LOG PLAN MAX_TOKENS
PLAN is a string like AAABBBABAB (A = Python code session, B = Chinese prose session)."""
import json, re, sys, time, urllib.request

port, elog, plan, maxtok = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
RX = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)(?:; (\d+) more read by the GPU over PCIe[^(]*\(([\d.]+)%)?")
ASKS = {
 "A": ["Write a Python module that implements an LRU cache with TTL expiry, with docstrings and type hints.",
       "Now add thread safety and a stats() method, and show the full module again.",
       "Write unittest tests covering expiry, eviction order and thread safety.",
       "Refactor it into a package with an async variant, and show every file.",
       "Add a command-line tool that loads a JSON file into the cache and prints stats, with argparse."],
 "B": ["请用中文写一篇关于宋代科举制度的长文，介绍它的起源、考试流程和对社会的影响。",
       "继续写：比较宋代和明代的科举，有哪些主要区别？请详细展开。",
       "请以一位落第书生的口吻，写一首七言律诗，并逐句解释。",
       "再写一个短篇故事：一个宋代女子如何在市井中经营茶馆，要有对话和细节。",
       "总结前面所有内容，并列出十个值得进一步研究的问题。"],
}
SYS = {"A": "You are a careful Python developer.", "B": "你是一位熟悉中国历史和古典文学的作家。"}
msgs = {k: [{"role": "system", "content": v}] for k, v in SYS.items()}
turn = {"A": 0, "B": 0}

def log_lines():
    with open(elog, errors="replace") as f:
        return f.readlines()

print("step sess turn  out  hit%   pcie%  lookups", flush=True)
for i, s in enumerate(plan):
    msgs[s].append({"role": "user", "content": ASKS[s][turn[s] % len(ASKS[s])]})
    n0 = len(log_lines())
    body = json.dumps({"model": "deepseek-v4.1-flash", "messages": msgs[s], "max_tokens": maxtok,
                       "temperature": 0.6}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json", "x-session-id": f"hitrate-{s}"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    msgs[s].append({"role": "assistant", "content": m.get("content") or ""})
    hit = None
    for _ in range(20):
        new = log_lines()[n0:]
        hits = [RX.search(l) for l in new if RX.search(l)]
        if hits: hit = hits[-1]; break
        time.sleep(0.25)
    out = r["usage"]["completion_tokens"]
    if hit:
        print(f"{i:4d}  {s}   {turn[s]:3d}  {out:4d}  {float(hit[1]):5.1f}  {float(hit[5] or 0):5.1f}  {hit[3]}", flush=True)
    else:
        print(f"{i:4d}  {s}   {turn[s]:3d}  {out:4d}  (no hit line)", flush=True)
    turn[s] += 1
