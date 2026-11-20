# DeepSeek UD-Q2_K_XL on Basalt, 2026-11-19

**Outcome:** Basalt runs our unmodified `DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL` file:
- **Quality:** within noise of Maya and llama.cpp, against BF16 and against the reference computation.
- **Generation:** 37% faster than Maya with its drafter, measured in the same session (91.2 vs 66.5 t/s warm median).
- **Long prompts:** Basalt reads them slower, taking 1.4-1.7x Maya's time to the first token.

Same machine as [2026-11-18-tuning.md](2026-11-18-tuning.md).
- **Basalt:** fork branch `feat/ds41-q2kxl` (`~/dev/basalt-ds41-q2kxl`, head `20ec37f`) of basalt-src `899e820`, plus
  the any-model-name patch in `serve/`, uncommitted for this run (see Upstream below). It adds:
  - Q5_K Engram rows.
  - MXFP4 down experts on the CPU and the GPU.
  - IQ3_S in the GPU down-format list.
- **Pack:** converted from the GGUF without requantization, 319.8 GiB, with Maya's DSpark drafter attached.
- **Maya:** v1.1.66 with its drafter (`llm-serve -d deepseek`).

## Quality (thresholds fixed before the runs)

| Check | Basalt | Reference points |
|---|---|---|
| Same weights vs `ref_gguf.py`, 128 tok (rows 64-126) | KL 0.00450, top-1 100% | Maya 0.0030 / 100%, llama.cpp 0.0196 / 98.4% |
| Same weights vs `ref_gguf.py`, 512 tok (rows 256-510) | KL 0.01491, top-1 97.65% | Maya 0.0181 / 97.3%, llama.cpp 0.0246 / 98.0% |
| vs BF16, first 20 chunks | KLD 0.13485, median 0.0071, top-1 91.27%, PPL 2.2962 | Maya 0.1366 / 91.00% / 2.2915 |
| vs BF16, 100 chunks (gate: <= 0.155, >= 88.0%) | **0.14936**, median 0.0196, p99 1.73, top-1 **88.58%**, PPL ratio 1.0025 | llama.cpp 0.1503 / 88.48% / 0.9994 |
| Placement: 168 vs 6 experts per layer on the GPU (20 chunks) | paired KLD diff +0.0019 +/- 0.0018 | - |
| Same run twice | bitwise identical on all 5100 rows | - |
| Planted fact in 9k / 32k-token prompts | recalled; first answer token p ~ 1 | Maya recalls it too |

Notes:
- **The two "same weights vs `ref_gguf.py`" rows can't rank Maya against Basalt.** The reference varies against itself
  by a mean KL of about 2e-3 (max about 1.1e-2): serial with different BLAS thread counts, and serial vs its parallel
  workers, on 32 tokens of real English. At 128 tokens both engines are within that max, and at 512 tokens both are
  just above it (0.015 and 0.018). The 2e-3 comes from one 32-token sentence, so treat it as an order of magnitude.
- **llama.cpp reference points** are its TF32-off run (`NVIDIA_TF32_OVERRIDE=0`, see
  [2026-11-18-tuning.md](2026-11-18-tuning.md)); with TF32 on it is 0.042 / 0.049.
- **Gate settings:** `basalt-bench --quality`, which runs the decode path with adaptation off and `--spec 5`. The lossy
  options `--route-tail-skip` and `--route-prior` are off by default.
- **CPU vs GPU experts round differently.** With most experts on the CPU, the rows are not bitwise equal to the default
  run: the top-1 token agrees 97.2% of the time. Quality is the same.
- **Greedy output is not bit-reproducible while serving, on either engine.** The same temperature-0 request repeated on
  one server changes text:
  - Basalt: from the 3rd repeat, as its GPU hit rate rises from 77% to 88% while it moves hot experts onto the GPU.
  - Maya: from the 2nd repeat.
  - So "drafter on vs off token-identical" can't be tested. The KLD gates cover the drafter instead.

## Speed (`bin/llm-bench --max-tokens 512 --long-words 5500 22000`, warm pass discarded)

| | Maya + drafter (rerun, same session) | **Basalt + drafter** |
|---|---|---|
| Gen, warm median (range) | 66.5 t/s (48.2-72.4) | **91.2 t/s (74.6-92.1)** |
| Gen, cold | 62.5 | 81.4 |
| Gen after 5.5k / 22k-word prompt | 54.4 / 63.9 | 76.3 / 82.8 |
| Short-prompt prefill, warm median | 36.1 t/s | 46.9 t/s |
| Time to first token, 7,555-token prompt | **4.3 s** | 6.2 s |
| Time to first token, 22k-word prompt | **7.5 s** (23,061 new tokens) | 12.8 s (all 30,104 tokens read) |

Notes:
- **Earlier Maya run:** today's first run gave 60.1 t/s (54.7-66.2). The rerun gave 66.5. Basalt's slowest prompt
  is still above Maya's median.
- **Long-prompt t/s doesn't compare.** Maya's prefix cache reused about 7k tokens of the 22k-word prompt. Basalt read
  it all. Time to first token is the fair comparison.
- **Same tokenizer output:** 7,555 tokens for the 5.5k-word prompt on both engines.
- **Raw data:** in `2026-11-19/`. In the serve lines, the first 8 lines are the discarded warm-up pass:
  - `basalt-ds41-q2kxl.json`, `vmstat-basalt-ds41-q2kxl.log`, `basalt-ds41-q2kxl-serve-lines.txt`
  - `maya-dspark-rerun.json`, `vmstat-maya-dspark-rerun.log`

## Multi-turn prompt reuse (`2026-11-19/multiturn.py`)

A 5,578-word document and five questions; each turn resends the whole conversation (10.9k-11.6k tokens), 300 tokens
out, temperature 0. Each engine was started cold, alone on the GPU, and answered the same turns.

| Turn | Basalt: cached / first token / gen | Maya + drafter: cached / first token / gen |
|---:|---|---|
| 1 | 0 / 8.7 s / 49.5 t/s | 0 / 8.0 s / 39.0 t/s |
| 2 | 10,873 / **1.2 s** / 67.6 t/s | 10,873 / 1.8 s / 52.3 t/s |
| 3 | 11,059 / **0.8 s** / 77.3 t/s | 11,113 / 1.3 s / 56.9 t/s |
| 4 | 11,075 / **0.8 s** / 68.3 t/s | 11,279 / 1.5 s / 53.8 t/s |
| 5 | 11,188 / **0.5 s** / 73.1 t/s | 11,444 / 1.6 s / 41.8 t/s |

- **Both engines reuse the conversation:** from turn 2 they read only the new tokens (98-99.8% cached).
- **One run of five 11k-token turns.** Gen t/s over 300 tokens is rough. Prompts of 100k+ tokens are not covered.
- **Build:** `~/dev/basalt-ds41-up`, the PR branch of jesdga95/basalt#481. The `llm-serve deepseek` profile runs
  the same code plus the any-model-name serve patch, from `~/dev/basalt-ds41-serve`.
- **Raw data:** `2026-11-19/multiturn-basalt.json`, `2026-11-19/multiturn-maya.json`.

## Terminal-Bench 2.0, 20 tasks (`2026-11-19/terminal-bench/`)

harbor 0.26.0, agent `terminus-2`, `-n 1`, `--agent-timeout-multiplier 2`, one attempt per task, through
`llm-serve -d deepseek` (Basalt) and `llm-serve -d deepseek-maya` (Maya + drafter) on :8000. Tasks:
`2026-11-19/tb-tasks.txt`.
Maya ran with its default 32,768-token thinking budget, Basalt with none. Only one agent step reached that size on
each engine, both on `torch-pipeline-parallelism`.

| | Maya + drafter | **Basalt + drafter** |
|---|---|---|
| Passed | **17 / 20** | 16 / 20 |
| Time, the 12 tasks both ran on a quiet machine | 185 min | **133 min** |
| Time, all 20 | not comparable (see below) | 2 h 56 min |

| Task | Maya | Basalt |
|---|---|---|
| `db-wal-recovery` | timed out (30.4 min) | **passed, 2.6 min** |
| `query-optimize` | failed, 15.1 min | **passed, 16.4 min** |
| `polyglot-c-py` | **passed, 10.4 min** | failed, 7.0 min |
| `polyglot-rust-c` | **passed, 13.0 min** (shared run) | failed, 3.8 min |
| `torch-pipeline-parallelism` | **passed** (agent hit its time limit, 33.1 min; the grader passed what it left) | failed, 21.8 min |
| `path-tracing-reverse` | timed out (60.6 min) | timed out (60.5 min) |
| the other 14 | passed | passed |

- **Maya's 20 come from two runs.** 12 tasks ran overnight on a quiet machine (`tb20-maya-b`), including reruns of
  `db-wal-recovery`, `path-tracing-reverse` and `torch-pipeline-parallelism`, whose first try shared the server.
  The other 8 come from the first pass (`tb20-maya`), which shared the server with other work. All 8 passed, so the pass
  counts stand, but their times are not comparable.
- **Both Basalt polyglot failures were on the grader's first check.** The agent left compiled binaries next to the
  source, and the grader requires the source to be the only file. It never got as far as testing the code.
- **No reasoning loops in any of the 43 trials.** In every reasoning trace of 2,000+ words, at most 16% of the 12-word phrases repeat,
  against the 90% that marks a loop below.
- **One attempt per task.** A one- or two-task gap is noise. What this shows: Basalt serves our UD-Q2_K_XL file through a
  real agent loop within one task of Maya's pass rate, and finishes faster.
- **Raw data:** `results.csv` (one row per trial), `night.sh` (the run), `summ.py`.

## Reasoning loops on long reviews

Two long code-review prompts (6.9k and 18.4k tokens, `max_tokens` 64000, temperature 0.6) came back empty from Basalt:
the model was still thinking at 64,000 tokens. Reruns found loops where 90% or more of the last 2,000 words' 12-word
phrases had already appeared.

- **Basalt, the 18.4k prompt, top_p 0.95, 4 runs each way:** drafter on, 3 answered and 1 looped; drafter off
  (`spec 0`), 3 answered and 1 looped. 1 in 4 looped either way, so there is no sign the drafter causes it (4 runs
  each is a small sample). Drafter on generated about 128 t/s against 72 t/s off.
- **Maya, same prompt and sampling, 4 runs with its default thinking budget:** all 4 answered, but each one hit the
  budget (32,768 tokens), where Maya closes the reasoning and makes the model answer. Maya's log says "thinking reached its
  budget" on all 4. These runs can't show whether Maya loops.
- **Maya, 4 runs with the budget off** (`POST /api/thinking-budget {"budget": 0}`, which matches Basalt's no-cap
  setup): 3 answered, at 32-36k tokens, and 1 looped (93.6% repeats at word 14,811). Two of the 3 that answered went
  through loop phases (78% and 84% repeats) and recovered.
- **Both engines loop, 1 run in 4 each, so the loops look like the model at these settings, not one engine** (small
  samples).
  Maya's default thinking budget is what keeps its long reviews from running into `max_tokens`. Basalt can do the same
  with `reasoning-budget-tokens` (a server default, 0 = none, which is our profile's setting) or a request's
  `reasoning_budget_tokens`. Its `thinking-loop-tokens` only catches one token repeated in a row, not phrase loops.

## Not done yet

- **Upstream:** the Basalt changes are proposed as [jesdga95/basalt#481](https://github.com/jesdga95/basalt/pull/481),
  rebased onto Basalt 0.5.0, with an independent review, parity tests and the KLD gate. The any-model-name serve patch
  was dropped: jesdga95/basalt#337 covers it.

