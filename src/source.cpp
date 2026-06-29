#include "llmtrace/source.hpp"

#include <string>

namespace llmtrace {

namespace {

// One step in a transformer block's forward pass. base_lat scales the simulated
// latency so attention/MLP matmuls dominate, as they do in reality.
struct Step {
    const char* suffix;
    const char* op;
    int shape_kind;   // 0: [d_model,seq]  1: [seq,seq,heads]  2: [d_ff,seq]  3: [d_model]
    double base_lat;
};

const Step kBlock[] = {
    {"attn_norm",     "RMS_NORM", 3, 0.05},
    {"Qcur",          "MUL_MAT",  0, 0.40},
    {"Kcur",          "MUL_MAT",  0, 0.40},
    {"Vcur",          "MUL_MAT",  0, 0.40},
    {"kq",            "MUL_MAT",  1, 0.70},
    {"kq_soft_max",   "SOFT_MAX", 1, 0.20},
    {"kqv",           "MUL_MAT",  1, 0.60},
    {"attn_out",      "MUL_MAT",  0, 0.45},
    {"ffn_norm",      "RMS_NORM", 3, 0.05},
    {"ffn_gate",      "MUL_MAT",  2, 0.90},
    {"ffn_up",        "MUL_MAT",  2, 0.90},
    {"ffn_down",      "MUL_MAT",  0, 0.95},
    {"l_out",         "ADD",      0, 0.06},
};
constexpr int kBlockSteps = static_cast<int>(sizeof(kBlock) / sizeof(kBlock[0]));

} // namespace

SyntheticSource::SyntheticSource(Config cfg) : cfg_(cfg), rng_(cfg.seed) {}

void SyntheticSource::advance() {
    ++step_;
    if (step_ >= kBlockSteps) {
        step_ = 0;
        ++layer_;
        if (layer_ >= cfg_.layers) {
            layer_ = 0;
            ++token_;
        }
    }
}

LayerEvent SyntheticSource::build(int layer, int step) {
    const Step& s = kBlock[step];
    std::uniform_real_distribution<double> jitter(0.8, 1.4);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    LayerEvent e;
    e.seq = seq_++;
    e.name = std::string(s.suffix) + "-" + std::to_string(layer);
    e.op = s.op;
    e.op_class = classify(e.name, &e.layer);
    e.layer = layer;

    switch (s.shape_kind) {
        case 0: e.shape = {cfg_.d_model, cfg_.seq_len, 1, 1}; e.n_dims = 2; break;
        case 1: e.shape = {cfg_.seq_len, cfg_.seq_len, cfg_.n_heads, 1}; e.n_dims = 3; break;
        case 2: e.shape = {cfg_.d_ff, cfg_.seq_len, 1, 1}; e.n_dims = 2; break;
        default: e.shape = {cfg_.d_model, 1, 1, 1}; e.n_dims = 1; break;
    }
    e.dtype = (s.op == std::string("SOFT_MAX")) ? "f32" : "f16";

    double lat = s.base_lat * jitter(rng_);
    // Rare CPU fallback for norms => latency spike + anomaly.
    bool fallback = (s.shape_kind == 3) && (u01(rng_) < 0.03f);
    e.device = fallback ? "CPU (fallback)" : "Metal";
    if (fallback) lat *= 6.0;
    e.latency_ms = lat;
    e.t_ms = t_ms_;
    t_ms_ += lat;
    last_pacing_ = lat;

    e.has_stats = true;
    e.mean = (u01(rng_) - 0.5f) * 0.4f;
    e.vmin = e.mean - (0.5f + u01(rng_));
    e.vmax = e.mean + (0.5f + u01(rng_) * 6.0f);
    e.sparsity = (e.op_class == OpClass::Mlp) ? 0.4f + u01(rng_) * 0.4f : u01(rng_) * 0.2f;

    if (e.vmax > 6.0f) {
        e.anomaly = true;
        e.anomaly_msg = "Outlier feature L" + std::to_string(layer) +
                        ": max " + std::to_string(e.vmax).substr(0, 4) + " > 6.0";
    } else if (fallback) {
        e.anomaly = true;
        e.anomaly_msg = "CUDA/Metal OOM fallback: " + std::string(s.suffix) +
                        " on CPU host memory (L" + std::to_string(layer) + ")";
    }
    return e;
}

bool SyntheticSource::next(LayerEvent& out) {
    if (!cfg_.loop && token_ >= cfg_.tokens) return false;
    out = build(layer_, step_);
    advance();
    return true;
}

ReplaySource::ReplaySource(std::vector<LayerEvent> events, bool loop)
    : events_(std::move(events)), loop_(loop) {}

bool ReplaySource::next(LayerEvent& out) {
    if (events_.empty()) return false;
    if (i_ >= events_.size()) {
        if (!loop_) return false;
        i_ = 0;
        last_t_ = 0.0;
    }
    out = events_[i_++];
    double dt = out.t_ms - last_t_;
    last_t_ = out.t_ms;
    last_pacing_ = (dt > 0.0 && dt < 1000.0) ? dt : out.latency_ms;
    return true;
}

} // namespace llmtrace
