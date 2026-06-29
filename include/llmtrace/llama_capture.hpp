#pragma once
#include <memory>
#include <string>

#include "llmtrace/source.hpp"

namespace llmtrace {

struct CaptureConfig {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int n_predict = 24;       // tokens to generate
    int n_ctx = 1024;
    int n_gpu_layers = 999;   // offload everything to Metal when available
    int seq_len = 16;         // attention window for the heatmap
};

// Drives a real llama.cpp inference run on a background thread and exposes the
// compute-graph nodes it observes — via ggml's eval callback — as a DataSource.
// Capture is non-invasive: we never modify model weights or model code, we only
// observe the tensors ggml hands us as the graph executes.
//
// next() blocks until the next event is available, and returns false once the run
// has finished and the queue is drained.
class LlamaCaptureSource : public DataSource {
public:
    explicit LlamaCaptureSource(CaptureConfig cfg);
    ~LlamaCaptureSource() override;

    bool next(LayerEvent& out) override;
    double pacing_ms() const override { return 0.0; }  // real-time; no artificial pacing

    bool ok() const;
    std::string error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmtrace
