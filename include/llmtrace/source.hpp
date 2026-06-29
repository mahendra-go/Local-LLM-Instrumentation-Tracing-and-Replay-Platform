#pragma once
#include <cstddef>
#include <random>
#include <vector>

#include "llmtrace/event.hpp"

namespace llmtrace {

// Abstract stream of capture events. The TUI is agnostic to where events come
// from: a synthetic generator, a recorded trace, or the live llama.cpp backend.
class DataSource {
public:
    virtual ~DataSource() = default;
    // Fill `out` with the next event; return false when the stream is exhausted.
    virtual bool next(LayerEvent& out) = 0;
    // Suggested delay before presenting the event just returned (live pacing).
    virtual double pacing_ms() const { return 0.0; }
};

// Generates a plausible per-token, per-layer op stream for a transformer, so the
// dashboard can be demoed without any model. Deterministic given a seed.
class SyntheticSource : public DataSource {
public:
    struct Config {
        int layers = 32;
        int tokens = 8;       // forward passes to simulate
        int n_heads = 32;
        int seq_len = 16;     // attention window shown in the heatmap
        int d_model = 4096;
        int d_ff = 11008;
        bool loop = true;     // keep streaming for a live dashboard
        unsigned seed = 1234;
    };

    explicit SyntheticSource(Config cfg);
    bool next(LayerEvent& out) override;
    double pacing_ms() const override { return last_pacing_; }

private:
    LayerEvent build(int layer, int step);
    void advance();

    Config cfg_;
    std::mt19937 rng_;
    std::uint64_t seq_ = 0;
    double t_ms_ = 0.0;
    double last_pacing_ = 0.0;
    int token_ = 0;
    int layer_ = 0;
    int step_ = 0;
};

// Replays a previously recorded trace, pacing by the recorded timestamps.
class ReplaySource : public DataSource {
public:
    explicit ReplaySource(std::vector<LayerEvent> events, bool loop = true);
    bool next(LayerEvent& out) override;
    double pacing_ms() const override { return last_pacing_; }
    std::size_t size() const { return events_.size(); }

private:
    std::vector<LayerEvent> events_;
    std::size_t i_ = 0;
    bool loop_;
    double last_t_ = 0.0;
    double last_pacing_ = 0.0;
};

} // namespace llmtrace
