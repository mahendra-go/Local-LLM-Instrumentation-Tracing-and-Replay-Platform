#pragma once
#include <array>
#include <cstdint>
#include <string>

namespace llmtrace {

// Coarse classification of a compute-graph node, derived non-invasively from the
// tensor name reported by ggml. Used to colour/group the dashboard.
enum class OpClass : std::uint8_t {
    Other = 0,
    Embedding,
    Attention,
    Mlp,
    Norm,
    Output,
    Rope,
};

const char* to_string(OpClass c);

// One observed compute-graph node. Captured once per node, per forward pass.
// Kept POD-ish and fixed-size so it lives happily in a ring buffer.
struct LayerEvent {
    std::uint64_t seq = 0;        // monotonic capture sequence id
    double t_ms = 0.0;            // ms since capture start
    double latency_ms = 0.0;      // wall time attributed to this node

    int layer = -1;               // transformer block index, -1 if not in a block
    OpClass op_class = OpClass::Other;
    std::string op;               // ggml op, e.g. "MUL_MAT", "SOFT_MAX"
    std::string name;             // tensor name, e.g. "kq-15", "ffn_out-3"

    std::array<std::int64_t, 4> shape{{0, 0, 0, 0}};
    int n_dims = 0;
    std::string dtype;            // "f32", "f16", "q4_K", ...
    std::string device;           // "CPU", "Metal", "CUDA0", ...

    // Optional runtime statistics, filled when tensor data is observed.
    bool has_stats = false;
    float mean = 0.0f;
    float vmin = 0.0f;
    float vmax = 0.0f;
    float sparsity = 0.0f;        // fraction of |x| < eps

    bool anomaly = false;
    std::string anomaly_msg;      // human-readable note when anomaly == true

    std::int64_t numel() const {
        std::int64_t n = 1;
        for (int i = 0; i < n_dims && i < 4; ++i) n *= shape[i];
        return n_dims ? n : 0;
    }
};

// Classify a tensor by name (and op) without touching the model. Also extracts the
// transformer block index into *out_layer when present (e.g. "ffn_out-15" -> 15).
OpClass classify(const std::string& name, int* out_layer = nullptr);

} // namespace llmtrace
