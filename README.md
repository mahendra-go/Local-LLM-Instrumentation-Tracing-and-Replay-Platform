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
dominates compute? Where numerical anomalies (clipping, outliers, OOM fallbacks) creep
in? `llmtrace` answers these by observing a real inference run, op by op, and surfacing
it in a `lazygit`/`btop`-style dashboard you navigate with the keyboard.

## How it stays non-invasive

The capture backend uses [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and
registers a **`ggml_backend_sched_eval_callback`**. ggml invokes this callback for every
node in the compute graph as it executes — so we read each tensor's name, op, shape,
dtype, device and time it, **without touching model weights or model code**. Events are
pushed into a fixed-size ring buffer so RAM stays flat regardless of run length.

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

```
[Tab]: Cycle Focus | [j/k]: Navigate | [Space]: Select | [+/-]: Contrast | [q]: Quit
┌ 1. MODEL TOPOLOGY ───────┐ ┌ 2. LIVE EVENT STREAM ─────────────────────┐
│ ▼ model                  │ │  ID  TIME         LAYER TYPE     DEVICE     │
│   ► embeddings           │ │  104 21:14:02.110  Attn (Self)   CPU        │
│   ▼ layers               │ │  105 21:14:02.114  MLP (SwiGLU)  CPU        │
│     ▶ layers.1 [active]  │ │  ...                                        │
└──────────────────────────┘ └────────────────────────────────────────────┘
┌ 3. ATTENTION MATRIX ─────────────────┐ ┌ 4. RUNTIME METRICS ──────────────┐
│ tokens × tokens heatmap (░▒▓█)       │ │ Shape [1,32,4096]  Dtype f16     │
│                                      │ │ Sparsity 54.2%  Latency 1.14 ms  │
└──────────────────────────────────────┘ └──────────────────────────────────┘
┌ 5. NUMERICAL ANOMALY LEDGER ─────────────────────────────────────────────┐
│ 21:14:02.114 ⚠ Outlier feature L0: max > 6.0                              │
└───────────────────────────────────────────────────────────────────────────┘
```

## Build

```bash
# core (TUI + synthetic/replay, no model needed)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# with the real llama.cpp capture backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLMTRACE_ENABLE_LLAMA=ON
cmake --build build -j
```

## Usage

```bash
# Live demo with synthetic data (no model required)
./build/llmtrace --synthetic --layers 32

# Capture a real run and record a trace
./build/llmtrace --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
                 --prompt "Explain attention in one sentence." \
                 --record traces/run.trace

# Replay a recorded trace into the dashboard
./build/llmtrace --replay traces/run.trace
```

## Status / roadmap

- [x] Project scaffold, CMake, ring buffer
- [x] Event model + topology builder
- [x] Trace format (record) + replay reader + synthetic source
- [x] FTXUI 5-panel dashboard with keyboard navigation
- [x] llama.cpp non-invasive capture backend (ggml eval callback)
- [ ] Bonus: sparsity/anomaly heuristics tuning, per-head attention, GPU device tags

## License

MIT — see [LICENSE](LICENSE).
