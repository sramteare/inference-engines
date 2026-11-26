# Fresh re-check of the two-card numbers (2026-11-26)

This is the same machine as [2026-11-25-links-and-expert-cache.md](2026-11-25-links-and-expert-cache.md): two RTX PRO 6000 Blackwell Max-Q cards on PCIe 5.0 x16, with 2 channels of DDR5-6400. Every number that the plan doc "Basalt: two cards as one pool" relies on was measured again here, with no other client using the servers. GPU 1 also drives the desktop; it was 21% and then 3% busy when the two link runs started. GB/s means 1e9 bytes per second.

How the speeds were measured:
- `bench/loadtest.py` reads every speed from the server's own `done` and `live` log lines, not from client timing.
- It drops any round in which the log shows a different number of finished requests than it sent, which usually means another client used the server. No round was dropped in the runs below.
- Prompts are fixed: a system prompt of filler words (2k or 50k tokens, different for each session) plus one of three Python coding tasks. Each round gives 1,500 output tokens per session at temperature 0.6, over 3 rounds.
- A round's total for a server is the median of the server's `live` total over the moments when all its sessions were generating. Each slot's live rate is that request's average so far, so the total leans slightly towards the slow start of each request. The pool total is the sum of those per-server totals in one round. The medians below are over rounds.
- Qwen is Qwen3.8-Flash-Next on llm-serve's `qwen` profile, with `PARALLEL=2` on :8000. DeepSeek is DeepSeek-V4.1-Flash on Basalt 0.5.6.

## Summary

| Measurement | 2026-11-25 | 2026-11-26 |
|---|---|---|
| Card to card copy, each way / both at once | 52 / 101 GB/s | 51.4–52.3 / 100.8–101.3 GB/s |
| GPU 0 reading GPU 1's memory from a kernel | 24.6 GB/s (GPU 0 was serving) | 51.4 GB/s |
| GPU 1 reading GPU 0's memory from a kernel | 50.2 GB/s | 45.9–47.6 GB/s |
| A card reading its own memory | 1,584–1,633 GB/s | 1,630–1,637 GB/s |
| RAM to one card / to both at once | 57 / 94 GB/s | 57.0 / 91.8–94.7 GB/s |
| Refilling 61.19 GiB of experts, one card / both | 1.2 s / 1.4 s | 1.2 s / 1.4 s |
| DeepSeek, one session (2k / 50k context) | 105–112 tok/s (not in a results file) | 118–127 / 113–132 tok/s |
| DeepSeek, one session on both cards | `basalt-bench` conv task, 16k prompt, GPU 0 as main: 121–130 tok/s, against 99–114 on one card | GPU 0 as main: 131–137 tok/s; GPU 1 as main: 126–145 (2k and 50k together), against 114–121 on one card in the same run |
| DeepSeek hit rate, two sessions taking turns | 92–97% | 92.0–96.9% |
| DeepSeek with the adaptive slots frozen: hit rate, code / prose | 70–74% / 40–48% | 70.4–74.1% / 40.4–47.1% |
| Same, generation speed, code / prose | not measured | 44–86 / 22–35 tok/s |
| DeepSeek with 98 of 384 experts per layer on SSD | 53–89 tok/s, 69–84% hit (not in a results file) | 66–91 tok/s (2k), 87–90 (50k); 80–90% hit |
| Qwen, one session (2k / 50k) | not measured on fixed prompts | 225–260 (median 247) / 232–237 (median 233) tok/s per request |
| Qwen, two sessions at once on one card, total (2k / 50k) | not measured on fixed prompts | 295–320 (median 318.5) / 206–282 (median 273) tok/s |
| Qwen, three sessions over two cards, total (2k / 50k) | not measured | 556–592 (median 556.5) / 508–541 (median 538) tok/s |

What changed:
- **Remote reads:** the slow "0 reads 1" figure from 2026-11-25 was load on GPU 0. On idle cards both directions run at 46–51 GB/s.
- **DeepSeek, one session:** faster than the 105–112 tok/s noted on 2026-11-25, which was not saved with its raw output.
- **Qwen:** on these fixed prompts one session runs at 225–260 tok/s with about 68% draft acceptance. In the same server's log (`llama-qwen.log` on 2026-11-26), the 691 requests from real coding work with at least 200 output tokens ran at 116–495 tok/s (median 291), some of them alongside other sessions. The faster half accepted more drafts (median 80% against 68%). A second slot on the same card adds 18–29% in total. Adding a second card for a third session gives about +75% at 2k and +97% at 50k over two sessions on one card (median 556.5 against 318.5, and 538 against 273).

## Links: `bench/gpulinks.cu --mib 1024`, then `--blob 11 --total 4096 --ring 1024`

Two runs, a few seconds apart. GPU 1 was 21% busy when run 1 started and 3% when run 2 started.

```
== run 1
copy 0->1                  median    52.34 GB/s   best    53.00 GB/s
copy 1->0                  median    51.44 GB/s   best    53.72 GB/s
copy 0<->1 both at once    median   101.27 GB/s   best   102.30 GB/s
read remote 0 reads 1      median    51.43 GB/s   best    51.74 GB/s
read remote 1 reads 0      median    45.86 GB/s   best    47.82 GB/s
read local 0               median  1637.16 GB/s   best  1642.09 GB/s
read local 1               median  1630.16 GB/s   best  1637.92 GB/s
h2d gpu0                   median    57.03 GB/s   best    57.03 GB/s
h2d gpu1                   median    56.95 GB/s   best    57.03 GB/s
h2d both at once           median    94.73 GB/s   best    95.65 GB/s
d2h gpu0                   median    56.50 GB/s   best    56.50 GB/s
d2h gpu1                   median    56.26 GB/s   best    56.39 GB/s
blob h2d gpu0              median    56.66 GB/s   best    56.67 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1              median    56.59 GB/s   best    56.62 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1 4 streams    median    56.40 GB/s   best    56.46 GB/s   refill 61.19 GiB: 1.2 s
blob h2d both at once      median    93.94 GB/s   best    95.69 GB/s   refill 61.19 GiB/card at half total rate: 1.4 s
blob peer 0->1             median    55.35 GB/s   best    55.49 GB/s   refill 61.19 GiB: 1.2 s
blob peer 1->0             median    55.05 GB/s   best    55.36 GB/s   refill 61.19 GiB: 1.2 s
== run 2
copy 0->1                  median    51.97 GB/s   best    53.74 GB/s
copy 1->0                  median    51.80 GB/s   best    52.55 GB/s
copy 0<->1 both at once    median   100.75 GB/s   best   101.93 GB/s
read remote 0 reads 1      median    51.39 GB/s   best    51.77 GB/s
read remote 1 reads 0      median    47.55 GB/s   best    50.42 GB/s
read local 0               median  1637.04 GB/s   best  1639.76 GB/s
read local 1               median  1632.50 GB/s   best  1635.29 GB/s
h2d gpu0                   median    57.03 GB/s   best    57.03 GB/s
h2d gpu1                   median    56.91 GB/s   best    56.97 GB/s
h2d both at once           median    91.81 GB/s   best    94.70 GB/s
d2h gpu0                   median    56.49 GB/s   best    56.50 GB/s
d2h gpu1                   median    56.30 GB/s   best    56.41 GB/s
blob h2d gpu0              median    56.67 GB/s   best    56.67 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1              median    56.60 GB/s   best    56.65 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1 4 streams    median    56.40 GB/s   best    56.60 GB/s   refill 61.19 GiB: 1.2 s
blob h2d both at once      median    90.88 GB/s   best    94.83 GB/s   refill 61.19 GiB/card at half total rate: 1.4 s
blob peer 0->1             median    55.35 GB/s   best    55.51 GB/s   refill 61.19 GiB: 1.2 s
blob peer 1->0             median    55.16 GB/s   best    55.46 GB/s   refill 61.19 GiB: 1.2 s
```

## DeepSeek, one session: `loadtest.py --target 8001:LOG:deepseek-v4.1-flash:1`

At startup the engine log (Basalt's `engine.log`) said:

```
basalt loader: the cards' share is read (70.34 GiB in 11.5 s); RAM next
basalt deepseek41: RAM expert tier 117.3 GiB (253 of 384 experts per layer) loaded in 23.7 s
basalt deepseek41: VRAM expert tier 158 of 384 experts per layer (131 beside the prompt workspaces)
basalt deepseek41: drafter on GPU 0: 2.41 GiB, 32 of 128 experts per stage, windows of at most 6 tokens (--spec 0: off)
basalt deepseek41: expert homes of 384 per layer: main card 131, second card 0, RAM 253, file 0; main card copies 0
```

The server runs with only GPU 1 visible, so the engine's "GPU 0" is physical GPU 1, and all 70.34 GiB is on that card. The VRAM tier holds 158 experts per layer: the 131 fixed homes plus 27 adaptive slots, which hold copies of experts whose home is in RAM. So homes add up to 131 + 253 = 384.

| Context | Round 1 | Round 2 | Round 3 | Hit rate |
|---|---|---|---|---|
| 2k | 127.4 tok/s | 118.2 | 121.1 | 93.4–95.6% |
| 50k | 125.4 | 131.5 | 113.1 | 94.3–95.5% |

Round 1 at each context processes the whole prompt; rounds 2 and 3 reuse it from the cache.

## DeepSeek on one card against both cards

This was measured later the same day, after a reboot, with Qwen stopped so that GPU 0 was free. Each setting was a fresh `llm-serve -d deepseek` on :8001 with the `GPUS` value shown, then `loadtest.py --target 8001:LOG:deepseek-v4.1-flash:1`, with the same prompts as above.

An `rsync` to a new HDD ran the whole time. It wrote 280–295 MB/s to the HDD, and read 50–295 MB/s from the NVMe, from which the model also loads. The avg10 of the `full` line in `/proc/pressure/io` was 17–44%. Once the model is loaded, generation reads nothing from disk ("file 0" below).

The one-card speeds fall within the earlier run's range (113.7–120.9 against 113.1–131.5 tok/s), but the 50k median is about 7% lower (116.2 against 125.4). So a small effect from the copy can't be ruled out. All three settings ran under the same copy, so the comparison between them holds.

| `GPUS` | Expert homes per layer: cards / RAM | 2k: per round, tok/s | 50k: per round | Hit rate | Read over PCIe |
|---|---|---|---|---|---|
| `1` (one card) | 133 / 251 | 120.2, 120.9, 119.3 (median 120.2) | 120.6, 116.2, 113.7 (median 116.2) | 94.6–96.0% | 1.6–2.5% |
| `0,1` (GPU 0 as main, the llm-serve default) | 196 / 188 | 131.1, 134.2, 136.1 (median 134.2) | 131.4, 137.1, 132.5 (median 132.5) | 96.3–99.9% | 23.4–27.3% |
| `1,0` (GPU 1 as main) | 200 / 184 | 126.3, 143.6, 144.3 (median 143.6) | 140.8, 145.0, 134.9 (median 140.8) | 96.8–99.9% | 25.1–27.9% |

From the engine log at startup:
- `GPUS=0,1`: "expert homes of 384 per layer: main card 0, second card 196, RAM 188, file 0; main card copies 170". The cards' share was 100.48 GiB.
- `GPUS=1,0`: "main card 0, second card 200, RAM 184, file 0; main card copies 165". The cards' share was 102.33 GiB.
- `GPUS=1`: the VRAM tier grew to 160 per layer, which is the 133 homes plus 27 adaptive slots, as in the earlier run.

What this means, from Basalt 0.5.6's source (`include/basalt/core/tier_policy.hpp` and `src/core/expert_tier.cpp`):
- **Where the homes are:** with two cards, every home in VRAM is on the second card (196–200 per layer), and the rest are in RAM (184–188).
- **What the main card holds:** copies of the most-used experts whose home is in RAM. An expert from RAM that misses twice within a short window is promoted into a main-card slot, and the least-used slot gives way.
- **The second card's homes can move:** experts can also be moved onto the second card over time.

The first round at each context reads the whole prompt; later rounds reuse it.

- **Both cards are faster:** +12% at 2k and +14% at 50k with GPU 0 as main, and +19% and +21% with GPU 1 as main, by median.
- **Why:** the server's `pcie=` share counts two things:
  - experts that the second card holds and computes itself. Only the activations and result rows cross PCIe, not the expert weights.
  - experts copied from RAM to a card.
  With both cards, this share is 23–28%, most of it work done on the second card. Of the lookups left to the main card, 99.4–99.9% hit its copies. On one card the share is 1.6–2.5%, all of it RAM-to-card copies, and 4–5% of lookups miss.
- **Which card leads:** GPU 1 as main was about 7% faster than GPU 0 as main. GPU 1 also drives the desktop. Why the order matters is not checked, and three rounds per setting is a small sample.
- **The 2026-11-25 comparison** (99–114 against 121–130 tok/s in `2026-11-25-two-gpus.md`) came from `basalt-bench` with different prompts, so it does not compare directly with these numbers.

## DeepSeek hit rate, two sessions taking turns: `hitrate.py 8001 ENGINE_LOG AAABBBABABAB 1000`

A is a Python coding conversation and B a Chinese prose one, 1,000 tokens per turn.

```
step sess turn  out  hit%   pcie%  lookups
   0  A     0  1000   95.7    1.7  228631
   1  A     1  1000   95.2    2.2  229462
   2  A     2  1000   95.1    2.3  213399
   3  B     0  1000   94.9    1.6  246634
   4  B     1  1000   96.9    0.8  245082
   5  B     2  1000   96.3    1.1  233240
   6  A     3  1000   92.5    3.9  201799
   7  B     3  1000   95.4    1.4  240521
   8  A     4  1000   92.0    4.8  187462
   9  B     4  1000   94.6    2.0  234293
  10  A     5  1000   92.4    4.1  203959
  11  B     5  1000   94.7    1.7  237779
```

Code turns after the first switch run at 92.0–92.5%, against 95.1–95.7% before it.

## DeepSeek with the adaptive slots frozen: `--diag-tier-frozen`, same `hitrate.py` plan

This turns off both the adaptive slots and reading experts over PCIe, so only the 131 fixed homes per layer are in VRAM. Speeds are from the server's `done` lines.

```
step sess turn  out  hit%   pcie%  lookups   tok/s
   0  A     0  1000   74.1    0.0  191388    86.4
   1  A     1  1000   74.0    0.0  225598    68.2
   2  A     2  1000   73.6    0.0  223083    68.5
   3  B     0  1000   44.9    0.0  233889    34.7
   4  B     1  1000   42.4    0.0  231707    33.9
   5  B     2  1000   40.4    0.0  216627    35.3
   6  A     3  1000   72.8    0.0  207936    70.7
   7  B     3  1000   47.1    0.0  228107    23.4
   8  A     4  1000   70.4    0.0  210045    43.7
   9  B     4  1000   45.4    0.0  233321    21.8
  10  A     5  1000   73.6    0.0  210758    47.0
  11  B     5  1000   45.9    0.0  230922    22.1
```

## DeepSeek with experts on SSD: `diag-tier-ram: 155`

Capping RAM at 155 experts per layer gave "expert homes of 384 per layer: main card 131, second card 0, RAM 155, file 98" and "RAM expert tier 71.9 GiB (155 of 384 experts per layer)" in the engine log: 131 + 155 + 98 = 384. The VRAM tier again held 158 per layer, 27 of them adaptive copies. Without the cap the same setup holds 253 in RAM and 0 on SSD.

| Context | Round 1 | Round 2 | Round 3 | Hit rate |
|---|---|---|---|---|
| 2k | 66.4 tok/s | 83.4 | 90.9 | 83.1–88.3% |
| 50k | 89.5 | 87.1 | 88.6 | 87.2–90.3% |

`hitrate.py` with the same plan gave 80.2–88.7% on every turn (the last turn stopped at 669 tokens).

## Qwen on one card: `loadtest.py --target 8000:LOG:qwen3.8-flash-next:N`

The total is the median of the server's `live` total while every session was generating.

| Context | Sessions | Per request, tok/s | Total, by round | Draft acceptance |
|---|---|---|---|---|
| 2k | 1 | 225.2, 247.4, 259.5 | 225, 247, 258.5 | 62–68% |
| 2k | 2 | 168.5–271.5 | 295, 318.5, 319.5 | 62–74% |
| 50k | 1 | 232.8, 236.9, 232.1 | 232, 235, 229 | 67–70% |
| 50k | 2 | 148.8–288.6 | 205.5, 273, 282 | 67–73% |

## Three Qwen sessions over two cards

This adds a second Qwen server on GPU 1: `GPUS=1 PORT=8002 LLM_SERVE_INSTANCE=gpu1 llm-serve -d qwen`. Two sessions go to :8000 and one goes to :8002: `loadtest.py --target 8000:LOG:qwen3.8-flash-next:2 --target 8002:LOG2:qwen3.8-flash-next:1`.

| Context | :8000 total, by round | :8002, by round | Pool total, by round |
|---|---|---|---|
| 2k | 306, 322.5, 325.5 | 249.5, 269, 231 | 555.5, 591.5, 556.5 (median 556.5) |
| 50k | 286.5, 309, 294 | 221, 232, 244 | 507.5, 541, 538 (median 538) |

The script printed "pool total 572" and "526" for these runs: that was the sum of each server's median, not the median over rounds, and it is fixed in this PR. On the same prompts, one card's two-session total plus a lone session (318.5 + 247 = 565.5 at 2k) is close to the measured median of 556.5.
