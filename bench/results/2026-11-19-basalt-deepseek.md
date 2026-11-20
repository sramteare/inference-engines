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

## Not done yet

- **Terminal-Bench:** 20 tasks are picked (`2026-11-19/tb-tasks.txt`). Maya's first pass is partial and some of its
  tasks shared the server with other work, so the Maya vs Basalt comparison is postponed to a quiet machine.
- **Upstream:** the Basalt changes are proposed as [jesdga95/basalt#481](https://github.com/jesdga95/basalt/pull/481),
  rebased onto Basalt 0.5.0, with an independent review, parity tests and the KLD gate. The any-model-name serve patch
  was dropped: jesdga95/basalt#337 covers it.
- **Prompt reuse across turns:** whether Basalt reuses a growing multi-turn prompt, as Maya's prefix cache does, is not
  measured yet.
