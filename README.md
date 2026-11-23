# inference-engines

Scripts, build recipes and benchmarks for running and tuning local LLM inference
engines (llama.cpp and vLLM) on one workstation. The engines themselves are not
committed: this repo records how to rebuild them exactly, how each model is
launched, and what the tuning runs measured.

## Machine

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (compute 12.0), driver 620.70 |
| CPU | AMD Ryzen Threadripper PRO 9955WX (16 cores) |
| RAM | 128 GB |
| CUDA | 13.5 (`/usr/local/cuda`) |

The llama.cpp build is native (`GGML_NATIVE=ON`, `CMAKE_CUDA_ARCHITECTURES=native`),
so its binaries are tuned to this CPU and GPU and are not portable.

## Layout

```
bin/llm-serve        start/stop/inspect llama-server profiles (qwen, deepseek)
bin/llm-stats        live metrics dashboard for a running llama-server
bin/llm-bench        reproducible benchmark suite (cold, warm, long context, MoE cache)
bin/llm-stop         stop llama-server instance(s)
build-llama.sh       check out the pinned llama.cpp tag and build it natively
setup-vllm.sh        recreate vllm-env/ from vllm/requirements.lock
vllm/                pinned vLLM environment
opencode/            opencode provider config matching the llm-serve profiles
bench/results/       benchmark runs worth keeping
```

Not tracked: `llama.cpp/` (clone of ggml-org/llama.cpp), `vllm-env/` (uv venv),
`logs/`. Models live outside the repo in `~/dev/models` (override with `MODELS=`).

## Setup

```bash
./build-llama.sh            # pinned tag (currently v1.0.0); pass another ref to try it
./setup-vllm.sh             # vLLM 0.33.0 + torch 2.14 (cu130)
ln -s "$PWD"/bin/llm-* ~/.local/bin/
```

## Serving

```bash
llm-serve qwen              # Qwen UD-Q4_K_XL on Basalt: faster, closer to BF16 (default)
llm-serve -d deepseek       # DeepSeek UD-Q2_K_XL on Basalt: same quality as Maya, ~37% faster generation, reuses multi-turn prompts, background
llm-serve deepseek-maya     # DeepSeek UD-Q2_K_XL on Project Maya: faster to the first token of a long new prompt
llm-serve qwen-llama        # Qwen3.8-Flash-Next UD-Q4_K_XL + MTP draft, 256k ctx (llama.cpp fallback)
llm-serve deepseek-llama    # DeepSeek-V4.1-Flash UD-Q2_K_XL + DSpark draft, 512k ctx (llama.cpp fallback)
llm-serve --status
llm-serve --stop
DRY_RUN=1 llm-serve qwen    # print the full launch command, start nothing
```

All profiles serve on port 8000, one at a time. `qwen` and `deepseek` run the
models on the Basalt engine and `deepseek-maya` on Project Maya instead of
llama.cpp (`qwen-basalt` and `deepseek-basalt` are kept as aliases of
`qwen` and `deepseek`); `llm-serve`
generates their config into `$LOG_DIR` (from
`$MAYA_DIR/maya-ds4.json`, minus the refusal-direction ablation args and the
`ablate` keys, for Maya) and `--status` / `--stop` cover them too: `--stop`
verifies each pid file still belongs to one of our servers (recorded start time
+ cmdline) before signalling anything, and stops servers by session, so engine
children go with them. Environment overrides: `PORT`, `HOST`,
`CTX`, `SPEC=0` (no speculative decoding), `MODELS`, `LLAMA_BIN`, `LOG_DIR`,
`MAYA_DIR` (default `~/dev/maya-src`),
`BASALT_DIR` (default `~/dev/engines/basalt-056`, Basalt 0.5.6),
`BASALT_DS_DIR` (default `~/dev/engines/basalt-ds41-056`; the Basalt build the
`deepseek` profile runs: 0.5.6 with the MXFP4/IQ3_S expert support of upstream
PR #481, which `BASALT_DIR` lacks), `BASALT_DS_PYTHON` (default
`$BASALT_DIR/.venv/bin/python`; the Python that runs the `BASALT_DS_DIR`
server; both default builds' `.venv` is a link to `~/dev/basalt-src/.venv`,
so that checkout must stay; the two build folders sit inside this checkout,
excluded only by the local `.git/info/exclude`, so never run `git clean -fdx`
here),
`DRY_RUN=1` (print the command — plus the generated config for the engine
profiles — and exit; no server, no opencode sync, nothing written to `$LOG_DIR`). Extra arguments after the
profile go to `llama-server` (or to `maya.py` / `basalt serve`).
`HOST` defaults to `0.0.0.0`, except `qwen`, `deepseek` and `deepseek-maya`
(the Basalt and Maya profiles),
which bind to `127.0.0.1` unless `HOST` is set explicitly.

The GPU also drives the desktop, so both profiles leave VRAM headroom:

| Env | Default | Effect |
|---|---|---|
| `VRAM_RESERVE_MIB` | 3072 | VRAM kept free for the desktop; qwen-llama passes it to `-fitt` |
| `MOE_CACHE_MIB` | computed | deepseek-llama MoE cache size; default = free VRAM (GPU 0) − `VRAM_RESERVE_MIB` − `DS_NONCACHE_MIB`, min 8000 (60000 if `nvidia-smi` fails) |
| `DS_NONCACHE_MIB` | 23000 | deepseek-llama's measured non-cache VRAM at 512k ctx with DSpark |
| `MOE_CPU_SHARE` | 0.85 | share of MoE cache misses computed on CPU from RAM experts instead of uploaded to GPU |

qwen-llama runs with f16 KV cache, `-ub 1024 -b 2048` and `-fitt $VRAM_RESERVE_MIB`;
deepseek-llama runs `-ngl 99 -cmoe`: the experts stay in RAM and a GPU cache holds the
most-used ones (`--moe-cache-mib`, sized from the free VRAM at startup); the CPU
computes most cache misses (`MOE_CPU_SHARE`).

### opencode

llama-server answers with whatever model is loaded, whatever name a request
uses. So that opencode always shows (and budgets context for) the loaded model,
`llm-serve` sets `model` and `small_model` in `~/.config/opencode/opencode.json`
to the profile's alias, and that model's context limit to the `-c` in use, every
time it starts a server. Each profile's `--alias` must have a matching entry
under the `llamacpp` provider: see `opencode/llamacpp-provider.json`.
Sessions already open keep their model until restarted. `OPENCODE_SYNC=0` turns
this off.

## Benchmarks

```bash
llm-bench --url http://127.0.0.1:8000 --out-md bench/results/$(date +%F)-<profile>-<change>.md
llm-bench --suite deepseek-ab        # managed A/B: -fit on vs -cmoe --moe-cache-mib
```

Record with each result the llama.cpp ref (`git -C llama.cpp describe --tags`),
the profile and any flags that differ from `llm-serve`, so a speedup can be
traced to a launch flag or a rebuild.

## License

MIT. See [LICENSE](LICENSE).
