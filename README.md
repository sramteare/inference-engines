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
llm-serve qwen              # Qwen3.8-Flash-Next UD-Q4_K_XL + MTP draft, 256k ctx
llm-serve -d deepseek       # DeepSeek-V4.1-Flash UD-Q2_K_XL + DSpark draft, 512k ctx, background
llm-serve --status
llm-serve --stop
```

Both profiles serve on port 8000, one at a time. Environment overrides: `PORT`,
`HOST`, `CTX`, `SPEC=0` (no speculative decoding), `MODELS`, `LLAMA_BIN`, `LOG_DIR`.
Extra arguments after the profile go straight to `llama-server`.

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
