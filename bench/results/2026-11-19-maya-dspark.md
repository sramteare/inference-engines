# Maya DSpark drafter A/B, 2026-11-19

**Outcome:** Maya's DSpark drafter makes DeepSeek generation 22% faster: the warm median goes from 49.1 to 60.1 t/s. It
costs 2-9% on long prompts and reads more from the SSD. It is lossless, because the full model checks every drafted
token. The `deepseek` pack now includes it, and Maya turns it on by default.

Same machine as [2026-11-18-tuning.md](2026-11-18-tuning.md): 1x RTX PRO 6000 Max-Q 96 GB, 122 GB RAM, Samsung 9100
PRO. Engine: Maya v1.1.66, model DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL, `llm-serve -d deepseek` on 127.0.0.1:8000.

## Results

| | Without drafter | With drafter |
|---|---|---|
| Gen, warm median | 49.1 t/s | **60.1 t/s (+22%)** |
| Gen, cold | 54.0 t/s | 60.6 t/s |
| Gen after 7.6k / 23k-token prompt | 46.1 / 52.6 t/s | 52.7 / 62.0 t/s |
| 7.6k / 23k-token prompt | 1,910 / 3,183 t/s | 1,741 / 3,128 t/s |
| Short-prompt prefill, warm median | 47.2 t/s | 34.9 t/s |
| Disk reads, average (peak) | 1.28 GB/s (4.9) | 1.85 GB/s (5.4) |
| iowait, average | 5.0% | 7.4% |
| Drafts kept | - | ~76-84% |

How it was measured:
- `bin/llm-bench --max-tokens 512 --long-words 5500 22000`, run twice per config. The first pass warmed the page cache
  and was discarded.
- `vmstat 5` ran during the measured pass. Raw JSON and the vmstat logs are in `2026-11-19/`.
- Before and after each pass, `/v1/models` confirmed that port 8000 was serving DeepSeek.
- Maya was the only engine on the GPU.

## Notes

- **The drafter's files:** `dspark/dspark.bin` (7.97 GB) and `dspark.json` come from
  `peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF` at revision `c33d87b8`, with the sha256 checked against Maya's
  `tools/ds4/download_gguf.json`. They hold DeepSeek's `mtp.*` blocks, which unsloth's GGUF leaves out.
- **Building the pack:** `tools/ds4/pack_gguf.py --model <dir> --drafter <dir>/dspark/dspark.json`. `--pack-info`
  reports 3 drafter blocks with 7.22 GB of experts.
- **Going back:** the pack without the drafter is kept at `maya-ds4.bak-nodspark`, next to the new pack.
- **Not the llama.cpp drafter:** `dspark-...-Q8_0.gguf` is for llama.cpp's `deepseek-llama` profile. Maya can't read
  that format.
- **SSD reads:** the drafter's experts compete with the main experts for RAM, so disk reads rise by about 45%. More RAM
  would remove most of these reads.
- **Start order matters:** start one engine only after the other has stopped. Basalt started while Maya still held VRAM
  came up with 13 GB and served only 61-71 t/s.
