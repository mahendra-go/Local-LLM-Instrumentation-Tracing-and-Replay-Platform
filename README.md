# llmtrace — Local LLM Instrumentation, Tracing & Replay Platform

A lightweight, **non-invasive** telemetry and diagnostic tool for local transformer
models. `llmtrace` hooks into the model's execution pipeline *without modifying the
model's code*, captures real-time intermediate state (tensor shapes, activation stats,
per-layer latency, attention) as tokens flow through the network, and presents it in an
interactive terminal UI (TUI) — with the ability to record traces and **replay** them
later.

> GDSC Open Projects Summer '26 — Problem Statement #1.

## Why

Ever wondered how data propagates through the layers of a transformer? Which block
dominates compute? Where numerical anomalies (clipping, outliers, NaN/Inf) creep in?
`llmtrace` answers these by observing a real inference run, op by op, and surfacing it
in a `lazygit`/`btop`-style dashboard you navigate with the keyboard.

## Quick demo (no model needed)

A recorded sample trace ships in the repo, so you can see the dashboard immediately:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/llmtrace --replay examples/demo.trace        # interactive dashboard
```

A text snapshot of the dashboard lives at [`docs/dashboard.txt`](docs/dashboard.txt).

## How it stays non-invasive

The capture backend uses [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and
registers a **`ggml_backend_sched_eval_callback`**. ggml invokes this callback for every
node in the compute graph as it executes — so we read each tensor's name, op, shape,
dtype, device, timing, and (for host-resident F32 tensors) **real activation statistics**,
*without touching model weights or model code*. Events go into a fixed-size ring buffer
so RAM stays flat regardless of run length.

```
                 model code (untouched)
                          │
              ggml compute graph executes
                          │  ggml_backend_sched_eval_callback(t, ask, user_data)
                          ▼
   ┌─────────────────────────────────────────┐
   │ Instrumentation  →  RingBuffer<LayerEvent>│ ──► .trace file (record)
   └─────────────────────────────────────────┘ ◄── .trace file (replay)
                          │
                          ▼
                  FTXUI dashboard (5 panels)
```

## Dashboard layout

Five panels (see [`docs/dashboard.txt`](docs/dashboard.txt) for a live snapshot):

1. **Model topology** — tree built live from the event stream; `j/k` to navigate,
   `Space` to expand/collapse, the active layer is tagged `[active]`.
2. **Live event stream** — every captured node: id, time, layer, type, device, tensor.
3. **Attention matrix** — causal heatmap (`+/-` adjusts contrast).
4. **Runtime metrics** — shape, dtype, device, latency, sparsity, mean/min/max for the
   selected layer's latest tensor.
5. **Numerical anomaly ledger** — outliers and NaN/Inf flagged as they occur.

`Tab` cycles panel focus, `q` quits.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. FTXUI is fetched automatically.

```bash
# core (TUI + synthetic/replay, no model needed)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# with the real llama.cpp capture backend (fetches & builds llama.cpp)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLMTRACE_ENABLE_LLAMA=ON
cmake --build build -j
```

On macOS: `brew install cmake`.

## Usage

```bash
# Live demo with synthetic data (no model required)
./build/llmtrace --synthetic --layers 32

# Replay the bundled sample (or any recorded trace)
./build/llmtrace --replay examples/demo.trace

# Capture a REAL run (requires -DLLMTRACE_ENABLE_LLAMA=ON):
./scripts/get-model.sh qwen        # downloads Qwen2.5-0.5B GGUF (~470 MB)
./build/llmtrace --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
                 --prompt "Explain attention in one sentence." --predict 16

# Record a real run to a trace, then replay it anywhere
./build/llmtrace --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
                 --prompt "Hello" --headless --record traces/run.trace
./build/llmtrace --replay traces/run.trace
```

Useful flags: `--layers N`, `--seq N` (attention window), `--predict N` (tokens),
`--ngl N` (GPU layers; `0` = CPU so F32 activations are host-readable for stats),
`--speed X` (replay pacing), `--selftest`.

## Verified

End-to-end on **Qwen2.5-0.5B-Instruct (Q4_K_M)**, CPU backend:

- **13,974** graph nodes captured across **24 layers** in a single short run
- real ops observed: `MUL_MAT`, `RMS_NORM`, `ROPE`, `FLASH_ATTN_EXT`, `GET_ROWS`, …
- real activation stats per F32 tensor; genuine outliers flagged
  (e.g. `Kcur` in layer 8 with `|val| 214.3`)
- `--selftest` covers the ring buffer, name classifier, and topology builder

## Project layout

```
include/llmtrace/   public headers (ring_buffer, event, topology, session, …)
src/                core, trace I/O, sources, session state, dashboard
src/capture/        llama.cpp capture backend (ggml eval callback)
scripts/get-model.sh   fetch a small GGUF model
examples/demo.trace    bundled sample trace (replay without a model)
docs/dashboard.txt     text snapshot of the dashboard
```

## Status

- [x] Project scaffold, CMake, ring buffer
- [x] Event model + topology builder
- [x] Trace format (record) + replay reader + synthetic source
- [x] FTXUI 5-panel dashboard with keyboard navigation
- [x] llama.cpp non-invasive capture backend (ggml eval callback) — verified
- [ ] Bonus: observed (not modelled) attention weights, per-head view, GPU device tags

## License

MIT — see [LICENSE](LICENSE).
