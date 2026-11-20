# Multi-turn prompt reuse: a document plus 5 questions, each turn resending the whole conversation.
# Prints per turn: prompt tokens, cached tokens (if the server reports them), time to first token, decode t/s.
import json, sys, time, urllib.request
url, doc_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
doc = open(doc_path).read()
qs = ["Summarize this document in five bullet points.",
      "Which command starts the server, and which flags matter most?",
      "What does the document say about model files and their layout?",
      "List the platforms mentioned and any limits stated for each.",
      "Write a short FAQ of three questions a new user would ask, with answers from the text."]
msgs = [{"role": "system", "content": "You answer questions about the document below.\n\n" + doc}]
rows = []
for i, q in enumerate(qs):
    msgs.append({"role": "user", "content": q})
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 300, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = None; ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if ttft is None: ttft = time.time() - t0
                    text += d.get("content") or ""; ntok += 1
    t1 = time.time()
    u = usage or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    comp = u.get("completion_tokens") or ntok
    row = {"turn": i + 1, "prompt_tokens": u.get("prompt_tokens"), "cached": cached, "ttft_s": round(ttft or -1, 2),
           "decode_tps": round(comp / max(t1 - t0 - (ttft or 0), 1e-6), 1), "completion": comp, "usage": u}
    rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "usage"}), flush=True)
    msgs.append({"role": "assistant", "content": text})
json.dump({"model": model, "url": url, "rows": rows}, open(out, "w"), indent=1)
