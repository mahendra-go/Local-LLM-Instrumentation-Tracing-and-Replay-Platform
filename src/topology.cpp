#include "llmtrace/topology.hpp"

namespace llmtrace {

Topology::Topology() {
    root_.label = "model";
    root_.children.push_back(TopoNode{"embeddings", -1, OpClass::Embedding, false, true, {}});
    root_.children.push_back(TopoNode{"layers", -1, OpClass::Other, false, true, {}});
    layers_group_ = root_.children.size() - 1;
    root_.children.push_back(TopoNode{"output", -1, OpClass::Output, false, true, {}});
}

TopoNode* Topology::ensure_layer(int layer) {
    auto& group = root_.children[layers_group_];
    auto it = layer_index_.find(layer);
    if (it != layer_index_.end()) {
        return &group.children[it->second];
    }
    TopoNode node;
    node.label = "layers." + std::to_string(layer);
    node.layer = layer;
    node.is_layer = true;
    node.expanded = false;  // collapsed by default to keep the tree compact
    // Insert keeping layers ordered by index.
    std::size_t pos = group.children.size();
    for (std::size_t i = 0; i < group.children.size(); ++i) {
        if (group.children[i].layer > layer) { pos = i; break; }
    }
    group.children.insert(group.children.begin() + pos, node);
    // Rebuild index map after insertion (indices shifted).
    layer_index_.clear();
    for (std::size_t i = 0; i < group.children.size(); ++i) {
        layer_index_[group.children[i].layer] = i;
    }
    if (layer + 1 > n_layers_) n_layers_ = layer + 1;
    return &group.children[layer_index_[layer]];
}

void Topology::observe(const LayerEvent& e) {
    if (e.layer < 0) return;
    TopoNode* ln = ensure_layer(e.layer);
    // Ensure a sub-block child for this op class exists under the layer.
    const char* sub = to_string(e.op_class);
    for (auto& c : ln->children) {
        if (c.label == sub) return;  // already present
    }
    ln->children.push_back(TopoNode{sub, e.layer, e.op_class, false, true, {}});
}

void Topology::flatten_into(const TopoNode& n, int depth, std::vector<Row>& out) const {
    out.push_back(Row{depth, &n, !n.children.empty()});
    if (n.expanded) {
        for (const auto& c : n.children) flatten_into(c, depth + 1, out);
    }
}

std::vector<Topology::Row> Topology::flatten() const {
    std::vector<Row> out;
    flatten_into(root_, 0, out);
    return out;
}

void Topology::move(int delta) {
    auto rows = flatten();
    if (rows.empty()) return;
    long s = static_cast<long>(sel_) + delta;
    if (s < 0) s = 0;
    if (s >= static_cast<long>(rows.size())) s = static_cast<long>(rows.size()) - 1;
    sel_ = static_cast<std::size_t>(s);
}

void Topology::toggle_selected() {
    auto rows = flatten();
    if (sel_ >= rows.size()) return;
    // Find the mutable node matching the selected (const) pointer by re-walking.
    const TopoNode* target = rows[sel_].node;
    // DFS over a mutable copy of the path: simplest is a recursive lambda over root_.
    struct Walk {
        const TopoNode* target;
        bool operator()(TopoNode& n) {
            if (&n == target) { n.expanded = !n.expanded; return true; }
            for (auto& c : n.children) if ((*this)(c)) return true;
            return false;
        }
    } walk{target};
    walk(root_);
}

int Topology::selected_layer() const {
    auto rows = flatten();
    if (sel_ >= rows.size()) return -1;
    return rows[sel_].node->layer;
}

} // namespace llmtrace
