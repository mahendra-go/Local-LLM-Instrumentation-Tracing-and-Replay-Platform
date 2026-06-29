#include "llmtrace/session.hpp"

#include <cmath>
#include <random>

namespace llmtrace {

SessionState::SessionState(std::size_t event_cap, std::size_t anomaly_cap, int seq_len)
    : events_(event_cap), anomalies_(anomaly_cap), seq_len_(seq_len > 0 ? seq_len : 16) {}

// Deterministic, plausible causal-attention pattern for a layer. We don't read the
// real attention tensor's values (that would mean copying activations every step);
// instead we visualise a stable per-layer pattern so the panel is meaningful in
// synthetic/replay mode. The live backend can later swap in observed values.
std::vector<std::vector<float>> SessionState::make_attention(int layer) const {
    const int n = seq_len_;
    std::vector<std::vector<float>> m(n, std::vector<float>(n, 0.0f));
    std::mt19937 rng(static_cast<unsigned>(layer * 2654435761u + 12345u));
    std::uniform_real_distribution<float> noise(0.0f, 1.0f);
    const float decay = 0.15f + 0.25f * noise(rng);  // varies per layer
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {  // causal mask
            float w = std::exp(-decay * (i - j)) * (0.6f + 0.4f * noise(rng));
            m[i][j] = w;
            sum += w;
        }
        if (sum > 0.0f) {
            for (int j = 0; j <= i; ++j) m[i][j] /= sum;  // row-normalise (softmax-like)
        }
    }
    return m;
}

void SessionState::ingest(const LayerEvent& e) {
    std::lock_guard<std::mutex> lk(m_);
    events_.push(e);
    if (e.anomaly) anomalies_.push(e);
    topo_.observe(e);
    total_latency_ += e.latency_ms;
    last_t_ms_ = e.t_ms + e.latency_ms;
    if (e.layer >= 0) {
        active_layer_ = e.layer;
        last_by_layer_[e.layer] = e;
        // Refresh the attention matrix when we cross the attention softmax of a layer.
        if (e.op_class == OpClass::Attention && e.name.find("soft_max") != std::string::npos) {
            attn_[e.layer] = make_attention(e.layer);
            last_attn_layer_ = e.layer;
        }
    }
}

std::vector<LayerEvent> SessionState::recent_events(std::size_t n) const {
    std::lock_guard<std::mutex> lk(m_);
    return events_.recent(n);
}

std::vector<LayerEvent> SessionState::recent_anomalies(std::size_t n) const {
    std::lock_guard<std::mutex> lk(m_);
    return anomalies_.recent(n);
}

bool SessionState::last_event_for_layer(int layer, LayerEvent& out) const {
    std::lock_guard<std::mutex> lk(m_);
    auto it = last_by_layer_.find(layer);
    if (it == last_by_layer_.end()) return false;
    out = it->second;
    return true;
}

std::vector<std::vector<float>> SessionState::attention(int preferred, int& out_layer) const {
    std::lock_guard<std::mutex> lk(m_);
    auto it = attn_.find(preferred);
    if (it != attn_.end()) { out_layer = preferred; return it->second; }
    if (last_attn_layer_ >= 0) {
        out_layer = last_attn_layer_;
        auto j = attn_.find(last_attn_layer_);
        if (j != attn_.end()) return j->second;
    }
    out_layer = -1;
    return {};
}

int SessionState::active_layer() const { std::lock_guard<std::mutex> lk(m_); return active_layer_; }
int SessionState::n_layers() const { std::lock_guard<std::mutex> lk(m_); return topo_.n_layers(); }
std::uint64_t SessionState::total_events() const { std::lock_guard<std::mutex> lk(m_); return events_.total(); }
double SessionState::total_latency_ms() const { std::lock_guard<std::mutex> lk(m_); return total_latency_; }

double SessionState::events_per_sec() const {
    std::lock_guard<std::mutex> lk(m_);
    if (last_t_ms_ <= 0.0) return 0.0;
    return static_cast<double>(events_.total()) / (last_t_ms_ / 1000.0);
}

std::vector<SessionState::TopoRow> SessionState::topo_rows() const {
    std::lock_guard<std::mutex> lk(m_);
    auto rows = topo_.flatten();
    std::size_t sel = topo_.selected();
    std::vector<TopoRow> out;
    out.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out.push_back(TopoRow{r.depth, r.node->label, r.node->layer, r.node->op_class,
                              r.has_children, r.node->expanded, i == sel, r.node->is_layer});
    }
    return out;
}

void SessionState::topo_move(int delta) { std::lock_guard<std::mutex> lk(m_); topo_.move(delta); }
void SessionState::topo_toggle() { std::lock_guard<std::mutex> lk(m_); topo_.toggle_selected(); }
int SessionState::topo_selected_layer() const { std::lock_guard<std::mutex> lk(m_); return topo_.selected_layer(); }

} // namespace llmtrace
