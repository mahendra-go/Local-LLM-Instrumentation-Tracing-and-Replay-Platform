#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "llmtrace/event.hpp"
#include "llmtrace/ring_buffer.hpp"
#include "llmtrace/topology.hpp"

namespace llmtrace {

// Thread-safe aggregate of everything the dashboard renders. A producer (synthetic,
// replay, or the live llama.cpp callback) calls ingest() from one thread while the
// UI thread reads snapshots — all guarded by a single mutex.
class SessionState {
public:
    SessionState(std::size_t event_cap, std::size_t anomaly_cap, int seq_len);

    // Fold one observed compute-graph node into all views.
    void ingest(const LayerEvent& e);

    // --- read-side snapshots (each locks and copies) ---
    std::vector<LayerEvent> recent_events(std::size_t n) const;
    std::vector<LayerEvent> recent_anomalies(std::size_t n) const;
    bool last_event_for_layer(int layer, LayerEvent& out) const;

    // Attention heatmap for `preferred` layer, else the most recently seen one.
    // Returns the matrix and reports which layer it belongs to via out_layer.
    std::vector<std::vector<float>> attention(int preferred, int& out_layer) const;

    int active_layer() const;          // layer of the most recent event
    int n_layers() const;
    std::uint64_t total_events() const;
    double total_latency_ms() const;
    double events_per_sec() const;

    // --- topology navigation (UI thread) ---
    struct TopoRow {
        int depth = 0;
        std::string label;
        int layer = -1;
        OpClass op_class = OpClass::Other;
        bool has_children = false;
        bool expanded = false;
        bool selected = false;
        bool is_layer = false;
    };
    std::vector<TopoRow> topo_rows() const;
    void topo_move(int delta);
    void topo_toggle();
    int topo_selected_layer() const;

private:
    std::vector<std::vector<float>> make_attention(int layer) const;

    mutable std::mutex m_;
    RingBuffer<LayerEvent> events_;
    RingBuffer<LayerEvent> anomalies_;
    Topology topo_;
    std::map<int, LayerEvent> last_by_layer_;
    std::map<int, std::vector<std::vector<float>>> attn_;
    int seq_len_;
    int active_layer_ = -1;
    int last_attn_layer_ = -1;
    double total_latency_ = 0.0;
    double last_t_ms_ = 0.0;
};

} // namespace llmtrace
