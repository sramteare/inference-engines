#!/usr/bin/env python3
"""Compare an engine's logits against a llama.cpp KL-divergence base file.

Base file format (written by llama-perplexity --kl-divergence-base), little-endian:
  magic "_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk,
  int32 tokens[n_chunk * n_ctx],
  then per chunk (n_ctx - 1 - n_ctx//2) rows of nv = 2*((n_vocab+1)//2)+4 uint16:
  first 4 uint16 are two float32 (scale, min_log_prob), then n_vocab uint16 q,
  base log-prob of token v is scale*q[v] + min_log_prob.

Scoring mirrors the log_softmax(n_vocab, logits, base_log_prob, tok, kld) overload
in llama.cpp tools/perplexity/perplexity.cpp.
"""

import argparse
import json
import os
import re
import struct
import sys
import tempfile

import numpy as np

MAGIC = b"_logits_"
F32 = np.float32
F64 = np.float64


def row_stride(n_vocab):
    return 2 * ((n_vocab + 1) // 2) + 4


def nearest_int(fval):
    # port of llama.cpp nearest_int(float)
    val = fval + F32(12582912.0)
    i = val.astype(F32).view(np.int32)
    return (i & 0x007FFFFF) - 0x00400000


def quantize_rows(logits, n_vocab):
    # port of the log_softmax(n_vocab, logits, log_prob, tok) writer, rows vectorized
    logits = np.ascontiguousarray(logits, F32)
    r = logits.shape[0]
    nv = row_stride(n_vocab)
    out = np.zeros((r, nv), np.uint16)
    max_logit = logits.max(axis=1)
    min_logit = logits.min(axis=1)
    min_logit = np.maximum(min_logit, max_logit - F32(16.0))
    sum_exp = np.exp(logits - max_logit[:, None]).astype(F64).sum(axis=1)
    log_sum_exp = np.log(sum_exp).astype(F32)
    min_log_prob = (min_logit - max_logit - log_sum_exp).astype(F32)
    scale = ((max_logit - min_logit) / F32(65535.0)).astype(F32)
    out[:, :4] = np.stack([scale, min_log_prob], axis=1).view(np.uint16)
    nonzero = scale != 0
    if np.any(nonzero):
        inv_scale = (F32(1.0) / np.where(nonzero, scale, F32(1.0))).astype(F32)
        d = (inv_scale[:, None] * (logits - min_logit[:, None])).astype(F32)
        q = np.where(logits > min_logit[:, None], nearest_int(d), 0)
        q = np.where(nonzero[:, None], q, 0).astype(np.uint16)
        out[:, 4:4 + n_vocab] = q
    return out


class BaseFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(8) != MAGIC:
                sys.exit(f"error: {path} is not a llama.cpp logits base file (bad magic)")
            n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
        if n_ctx < 2 or n_vocab < 1 or n_chunk < 1:
            sys.exit(f"error: bad header in {path}: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}")
        self.n_ctx = n_ctx
        self.n_vocab = n_vocab
        self.n_chunk = n_chunk
        self.first = n_ctx // 2
        self.rows_per_chunk = n_ctx - 1 - self.first
        self.nv = row_stride(n_vocab)
        self.tokens_off = 20
        self.data_off = self.tokens_off + 4 * n_chunk * n_ctx
        need = self.data_off + n_chunk * self.rows_per_chunk * self.nv * 2
        have = os.path.getsize(path)
        if have < need:
            sys.exit(f"error: {path} truncated: {have} bytes, need {need}")
        with open(path, "rb") as f:
            f.seek(self.tokens_off)
            self.tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype="<i4")

    def chunk_rows(self, c):
        offset = self.data_off + c * self.rows_per_chunk * self.nv * 2
        return np.memmap(self.path, dtype="<u2", mode="r", offset=offset,
                         shape=(self.rows_per_chunk, self.nv))

    def chunk_tokens(self, c):
        return self.tokens[c * self.n_ctx:(c + 1) * self.n_ctx]


def score_rows(test, base, toks, n_vocab):
    # test: (R, n_vocab) float32; base: (R, nv) uint16; toks: (R,) int; mirror llama.cpp KL reader
    test = np.ascontiguousarray(test, F32)
    r = test.shape[0]
    sm = base[:, :4].copy().view(F32)
    scale = sm[:, 0]
    min_log_prob = sm[:, 1]
    q = base[:, 4:4 + n_vocab]

    max_logit = test.max(axis=1)
    imax = test.argmax(axis=1)
    sum_exp = np.exp(test - max_logit[:, None]).astype(F64).sum(axis=1)
    log_sum_exp = np.log(sum_exp).astype(F32)
    lse_total = (max_logit + log_sum_exp).astype(F32)

    nll = (lse_total - test[np.arange(r), toks]).astype(F32)
    nll_base = (-(scale * q[np.arange(r), toks].astype(F32) + min_log_prob)).astype(F32)

    p_log_base = scale[:, None] * q.astype(F32) + min_log_prob[:, None]
    imax_base = p_log_base.argmax(axis=1)
    inner = (p_log_base - test) + lse_total[:, None]
    contrib = np.where(p_log_base > F32(-16.0), np.exp(p_log_base) * inner, F32(0.0))
    kld = contrib.astype(F64).sum(axis=1)
    same_top = imax == imax_base
    p_diff = (np.exp(-nll) - np.exp(-nll_base)).astype(F32)

    return {
        "kld": kld,
        "nll": nll.astype(F64),
        "nll_base": nll_base.astype(F64),
        "same_top": same_top,
        "p_diff": p_diff.astype(F64),
    }


def aggregate(rows):
    def cat(key):
        return np.concatenate(rows[key]) if rows[key] else np.zeros(0)
    kld = cat("kld")
    nll = cat("nll")
    nll_base = cat("nll_base")
    p_diff = cat("p_diff")
    count = len(kld)
    st = {
        "rows": int(count),
        "ppl_test": float(np.exp(nll.sum() / count)) if count else None,
        "ppl_base": float(np.exp(nll_base.sum() / count)) if count else None,
        "kld_mean": float(kld.mean()) if count else None,
        "kld_median": float(np.median(kld)) if count else None,
        "kld_p99": float(np.percentile(kld, 99)) if count else None,
        "kld_max": float(kld.max()) if count else None,
        "same_top_pct": float(100.0 * cat("same_top").mean()) if count else None,
        "rms_dp_pct": float(100.0 * np.sqrt((p_diff ** 2).mean())) if count else None,
    }
    return st


def new_acc():
    return {"kld": [], "nll": [], "nll_base": [], "same_top": [], "p_diff": []}


def fmt_stats(st):
    def num(v, spec):
        return format(v, spec) if v is not None else "n/a"
    return (f"rows={st['rows']:6d}  PPL(test)={num(st['ppl_test'], '10.4f')}  PPL(base)={num(st['ppl_base'], '10.4f')}  "
            f"KLD mean={num(st['kld_mean'], '10.5f')} median={num(st['kld_median'], '10.5f')} "
            f"p99={num(st['kld_p99'], '10.5f')} max={num(st['kld_max'], '10.5f')}  "
            f"same-top={num(st['same_top_pct'], '6.2f')}%  RMS dp={num(st['rms_dp_pct'], '6.3f')}%")


BLOCK = 64


def cmd_tokens(args):
    bf = BaseFile(args.base)
    k = min(args.chunks, bf.n_chunk)
    if k < args.chunks:
        print(f"warning: only {bf.n_chunk} chunks in base file, using {k}", file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    for c in range(k):
        toks = bf.chunk_tokens(c).astype(np.int64).copy()
        toks[0] = args.bos
        path = os.path.join(args.out, f"chunk-{c}.ids")
        with open(path, "w") as f:
            f.write(",".join(str(int(t)) for t in toks))
            f.write("\n")
    print(f"n_ctx    = {bf.n_ctx}")
    print(f"n_vocab  = {bf.n_vocab}")
    print(f"n_chunk  = {bf.n_chunk}")
    print(f"wrote {k} chunk files to {args.out}")


def cmd_kld_to_dumps(args):
    bf = BaseFile(args.base)
    k = min(args.chunks, bf.n_chunk)
    if k < args.chunks:
        print(f"warning: only {bf.n_chunk} chunks in base file, using {k}", file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    for c in range(k):
        base = bf.chunk_rows(c)
        path = os.path.join(args.out, f"chunk-{c}.f32")
        with open(path, "wb") as f:
            for b0 in range(0, bf.first, BLOCK):
                n = min(BLOCK, bf.first - b0)
                f.write(np.zeros((n, bf.n_vocab), F32).tobytes())
            for b0 in range(0, bf.rows_per_chunk, BLOCK):
                blk = np.asarray(base[b0:min(b0 + BLOCK, bf.rows_per_chunk)])
                sm = blk[:, :4].copy().view(F32)
                scale = sm[:, 0:1]
                min_log_prob = sm[:, 1:2]
                q = blk[:, 4:4 + bf.n_vocab].astype(F32)
                rows = np.where(q > 0, scale * q + min_log_prob, min_log_prob - F32(30.0))
                f.write(np.ascontiguousarray(rows, F32).tobytes())
            f.write(np.zeros((1, bf.n_vocab), F32).tobytes())
        del base
    print(f"n_ctx    = {bf.n_ctx}")
    print(f"n_vocab  = {bf.n_vocab}")
    print(f"n_chunk  = {bf.n_chunk}")
    print(f"wrote {k} dump files to {args.out}")


def compare_chunks(bf, dumps_dir, k):
    chunk_results = []
    all_rows = new_acc()
    for c in range(k):
        path = os.path.join(dumps_dir, f"chunk-{c}.f32")
        if not os.path.isfile(path):
            sys.exit(f"error: missing dump file {path}")
        size = os.path.getsize(path)
        if size % (4 * bf.n_ctx) != 0:
            sys.exit(f"error: {path} size {size} is not a multiple of n_ctx*4 = {4 * bf.n_ctx}")
        n_vocab_maya = size // (4 * bf.n_ctx)
        if n_vocab_maya < bf.n_vocab:
            sys.exit(f"error: {path} has {n_vocab_maya} columns, fewer than base n_vocab {bf.n_vocab}")
        dump = np.memmap(path, dtype="<f4", mode="r", shape=(bf.n_ctx, n_vocab_maya))
        base = bf.chunk_rows(c)
        tt = bf.chunk_tokens(c)

        extra_rows = 0
        if n_vocab_maya > bf.n_vocab:
            for b0 in range(0, bf.n_ctx, BLOCK):
                b1 = min(b0 + BLOCK, bf.n_ctx)
                blk = dump[b0:b1]
                smax = blk[:, :bf.n_vocab].max(axis=1)
                emax = blk[:, bf.n_vocab:].max(axis=1)
                extra_rows += int(np.count_nonzero(emax > smax))
            if extra_rows:
                print(f"WARNING: {path}: {extra_rows} rows have a logit outside the shared "
                      f"vocabulary larger than the max inside it", file=sys.stderr)

        rows = new_acc()
        for b0 in range(0, bf.rows_per_chunk, BLOCK):
            b1 = min(b0 + BLOCK, bf.rows_per_chunk)
            test = np.asarray(dump[bf.first + b0:bf.first + b1, :bf.n_vocab])
            r = score_rows(test, np.asarray(base[b0:b1]), tt[bf.first + b0 + 1:bf.first + b1 + 1],
                           bf.n_vocab)
            for key in rows:
                rows[key].append(r[key])
        st = aggregate(rows)
        st["chunk"] = c
        st["extra_vocab_rows"] = extra_rows
        chunk_results.append(st)
        for key in all_rows:
            all_rows[key].extend(rows[key])
        del dump, base
    return chunk_results, all_rows


def cmd_compare(args):
    bf = BaseFile(args.base)
    k = min(args.chunks, bf.n_chunk)
    if k < args.chunks:
        print(f"warning: only {bf.n_chunk} chunks in base file, using {k}", file=sys.stderr)
    chunk_results, all_rows = compare_chunks(bf, args.dumps, k)
    print(f"base={args.base}  n_ctx={bf.n_ctx} n_vocab={bf.n_vocab} n_chunk={bf.n_chunk}")
    for st in chunk_results:
        print(f"chunk {st['chunk']:3d}  {fmt_stats(st)}")
    overall = aggregate(all_rows)
    print(f"overall   {fmt_stats(overall)}")
    result = {
        "base": args.base,
        "n_ctx": bf.n_ctx,
        "n_vocab": bf.n_vocab,
        "n_chunk": bf.n_chunk,
        "chunks_scored": k,
        "per_chunk": chunk_results,
        "overall": overall,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json}")


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def cmd_lastrow(args):
    a_files = {n for n in os.listdir(args.a_dir) if os.path.isfile(os.path.join(args.a_dir, n))}
    b_files = {n for n in os.listdir(args.b_dir) if os.path.isfile(os.path.join(args.b_dir, n))}
    common = sorted(a_files & b_files, key=natural_key)
    if not common:
        sys.exit(f"error: no file name present in both {args.a_dir} and {args.b_dir}")
    results = []
    for name in common:
        a = np.fromfile(os.path.join(args.a_dir, name), dtype="<f4")
        b = np.fromfile(os.path.join(args.b_dir, name), dtype="<f4")
        if a.size == 0 or b.size == 0:
            print(f"{name}: empty, skipped", file=sys.stderr)
            continue
        if a.size != b.size:
            print(f"{name}: size mismatch ({a.size} vs {b.size}), skipped", file=sys.stderr)
            continue

        def log_softmax(x):
            x = x.astype(F64)
            m = x.max()
            return x - m - np.log(np.exp(x - m).sum())

        la = log_softmax(a)
        lb = log_softmax(b)
        kl = float((np.exp(la) * (la - lb)).sum())
        amax_a = int(np.argmax(a))
        amax_b = int(np.argmax(b))
        max_diff = float(np.abs(a.astype(F64) - b.astype(F64)).max())
        res = {
            "file": name,
            "n": int(a.size),
            "kl": kl,
            "argmax_a": amax_a,
            "argmax_b": amax_b,
            "argmax_same": amax_a == amax_b,
            "max_abs_logit_diff": max_diff,
        }
        results.append(res)
        print(f"{name:20s}  n={res['n']:8d}  KL(A||B)={kl:12.6f}  argmax {amax_a} vs {amax_b} "
              f"{'SAME' if res['argmax_same'] else 'DIFF'}  max|dlogit|={max_diff:8.4f}")
    if not results:
        sys.exit("error: no comparable file pairs: every file name present in both directories "
                 "is empty or size-mismatched")
    kls = [r["kl"] for r in results]
    n_same = sum(1 for r in results if r["argmax_same"])
    summary = {
        "files": len(results),
        "kl_mean": float(np.mean(kls)),
        "kl_max": float(np.max(kls)),
        "argmax_same": n_same,
        "argmax_total": len(results),
        "argmax_same_pct": float(100.0 * n_same / len(results)),
    }
    print(f"summary: files={summary['files']}  KL mean={summary['kl_mean']:.6f} "
          f"max={summary['kl_max']:.6f}  argmax agree {n_same}/{len(results)} "
          f"({summary['argmax_same_pct']:.1f}%)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"per_file": results, "summary": summary}, f, indent=2)
        print(f"wrote {args.json}")


def cmd_seqs(args):
    with open(args.tokens_file) as f:
        text = f.read()
    toks = [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]
    need = args.count * (args.len - 1)
    if len(toks) < need:
        sys.exit(f"error: need {need} tokens for {args.count} sequences of length {args.len}, "
                 f"file has {len(toks)}")
    os.makedirs(args.out, exist_ok=True)
    for kk in range(args.count):
        stretch = toks[kk * (args.len - 1):(kk + 1) * (args.len - 1)]
        line = ",".join(str(x) for x in [args.bos] + stretch)
        path = os.path.join(args.out, f"seq-{kk}.ids")
        with open(path, "w") as f:
            f.write(line + "\n")
    print(f"wrote {args.count} sequences of length {args.len} to {args.out} "
          f"(consumed {need} of {len(toks)} tokens)")


# ---------------------------------------------------------------- selftest

def write_base_file(path, n_ctx, n_vocab, tokens, logits):
    # tokens: (n_chunk, n_ctx) int32; logits: (n_chunk, rows_per_chunk, n_vocab) float32
    n_chunk = tokens.shape[0]
    first = n_ctx // 2
    rpc = n_ctx - 1 - first
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.astype("<i4").tobytes())
        for c in range(n_chunk):
            f.write(quantize_rows(logits[c], n_vocab).tobytes())
    assert tokens.shape == (n_chunk, n_ctx)
    assert logits.shape == (n_chunk, rpc, n_vocab)


def run_selftest():
    n_ctx = 8
    n_vocab = 10
    n_chunk = 2
    first = n_ctx // 2
    rpc = n_ctx - 1 - first
    rng = np.random.default_rng(42)

    logits = (rng.normal(size=(n_chunk, rpc, n_vocab)) * 3.0).astype(F32)
    tokens = rng.integers(0, n_vocab, size=(n_chunk, n_ctx)).astype(np.int32)

    dump_a = rng.normal(size=(n_chunk, n_ctx, n_vocab)).astype(F32)
    dump_a[:, first:first + rpc, :] = logits
    dump_b = dump_a.copy()
    dump_b[:, first:first + rpc, :] += (rng.normal(size=(n_chunk, rpc, n_vocab))).astype(F32)

    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "base.bin")
        write_base_file(base_path, n_ctx, n_vocab, tokens, logits)
        bf = BaseFile(base_path)
        results = {}
        for name, dump in (("identical", dump_a), ("noisy", dump_b)):
            ddir = os.path.join(tmp, name)
            os.makedirs(ddir)
            for c in range(n_chunk):
                dump[c].astype("<f4").tofile(os.path.join(ddir, f"chunk-{c}.f32"))
            chunk_results, all_rows = compare_chunks(bf, ddir, n_chunk)
            results[name] = aggregate(all_rows)

        rt_out = os.path.join(tmp, "roundtrip")

        class R:
            base = base_path
            chunks = n_chunk
            out = rt_out
        cmd_kld_to_dumps(R())
        chunk_results, all_rows = compare_chunks(bf, rt_out, n_chunk)
        results["roundtrip"] = aggregate(all_rows)

        tok_out = os.path.join(tmp, "ids")

        class A:
            base = base_path
            bos = 999
            chunks = n_chunk
            out = tok_out
        cmd_tokens(A())
        line = open(os.path.join(tok_out, "chunk-0.ids")).read().strip()
        got = [int(x) for x in line.split(",")]
        want = [int(t) for t in tokens[0]]
        want[0] = 999
        tokens_ok = got == want

    ident = results["identical"]
    noisy = results["noisy"]
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    print("selftest: identical dumps")
    print(f"  {fmt_stats(ident)}")
    check(ident["rows"] == 2 * rpc, f"scored {2 * rpc} rows")
    check(abs(ident["kld_mean"]) < 1e-4, f"KLD mean {ident['kld_mean']:.2e} < 1e-4 (uint16 rounding)")
    check(ident["kld_max"] < 1e-3, f"KLD max {ident['kld_max']:.2e} < 1e-3 (uint16 rounding)")
    check(ident["same_top_pct"] == 100.0, "same top-1 100%")
    check(abs(ident["ppl_test"] - ident["ppl_base"]) < 0.01,
          f"PPL test {ident['ppl_test']:.6f} ~= base {ident['ppl_base']:.6f}")

    print("selftest: noisy dumps")
    print(f"  {fmt_stats(noisy)}")
    check(noisy["kld_mean"] > 1e-2, f"KLD mean {noisy['kld_mean']:.4f} > 1e-2")
    check(noisy["kld_max"] > noisy["kld_mean"], "max KLD > mean KLD")

    print("selftest: tokens command")
    check(tokens_ok, "chunk-0.ids matches base tokens with BOS replacement")

    rt = results["roundtrip"]
    print("selftest: kld-to-dumps round trip")
    print(f"  {fmt_stats(rt)}")
    check(rt["rows"] == 2 * rpc, f"scored {2 * rpc} rows")
    check(rt["kld_mean"] < 1e-1,
          f"KLD mean {rt['kld_mean']:.2e} < 1e-1 (only the q==0 -30 shift biases it: with a 10-token "
          f"synthetic vocab min_log_prob > -16, real vocab cuts those rows off at -16 -> KLD ~ 0)")
    check(rt["kld_mean"] < noisy["kld_mean"] / 2,
          f"round-trip KLD {rt['kld_mean']:.4f} well below noisy dumps {noisy['kld_mean']:.4f}")
    check(rt["same_top_pct"] == 100.0, "same top-1 100%")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("tokens", help="write the per-chunk token files llama.cpp fed (BOS-substituted)")
    sp.add_argument("base", help="llama.cpp KLD base file")
    sp.add_argument("--bos", type=int, required=True, help="BOS token id")
    sp.add_argument("--chunks", type=int, required=True, help="write chunks c < K")
    sp.add_argument("--out", required=True, help="output directory")
    sp.set_defaults(func=cmd_tokens)

    sp = sub.add_parser("compare", help="score Maya logits dumps against the KLD base file")
    sp.add_argument("base", help="llama.cpp KLD base file")
    sp.add_argument("--dumps", required=True, help="directory with chunk-<c>.f32 dumps")
    sp.add_argument("--chunks", type=int, required=True, help="score chunks c < K")
    sp.add_argument("--json", help="optional JSON output path")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("kld-to-dumps", help="convert a KLD base file to float32 logit dumps for the compare command")
    sp.add_argument("base", help="llama.cpp KLD base file")
    sp.add_argument("--chunks", type=int, required=True, help="write chunks c < K")
    sp.add_argument("--out", required=True, help="output directory")
    sp.set_defaults(func=cmd_kld_to_dumps)

    sp = sub.add_parser("lastrow", help="compare single-row last-position logits between two engines")
    sp.add_argument("a_dir", help="directory of float32 row files (engine A)")
    sp.add_argument("b_dir", help="directory of float32 row files (engine B)")
    sp.add_argument("--json", help="optional JSON output path")
    sp.set_defaults(func=cmd_lastrow)

    sp = sub.add_parser("seqs", help="split a token id file into non-overlapping BOS-prefixed sequences")
    sp.add_argument("--tokens-file", required=True, help="file of comma/whitespace separated token ids")
    sp.add_argument("--len", type=int, required=True, help="sequence length (BOS + L-1 tokens)")
    sp.add_argument("--count", type=int, required=True, help="number of sequences")
    sp.add_argument("--bos", type=int, required=True, help="BOS token id")
    sp.add_argument("--out", required=True, help="output directory")
    sp.set_defaults(func=cmd_seqs)

    sp = sub.add_parser("selftest", help="self-test with a synthetic base file (no model needed)")
    sp.set_defaults(func=lambda args: sys.exit(run_selftest()))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
