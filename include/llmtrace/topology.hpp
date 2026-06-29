#pragma once
#include <map>
#include <string>
#include <vector>

#include "llmtrace/event.hpp"

namespace llmtrace {

// A node in the model topology tree shown in panel 1. The tree is built
// incrementally from the observed event stream (we never read the model file).
struct TopoNode {
    std::string label;
    int layer = -1;                 // -1 for structural nodes (root/groups)
    OpClass op_class = OpClass::Other;
    bool is_layer = false;          // true for "layers.N" rows
    bool expanded = true;
    std::vector<TopoNode> children;
};

// Builds and owns the topology tree, and provides a flattened, selection-aware
// view for rendering.
class Topology {
public:
    Topology();

    // Fold a freshly observed event into the tree (adds layers/blocks as seen).
    void observe(const LayerEvent& e);

    int n_layers() const { return n_layers_; }

    struct Row {
        int depth = 0;
        const TopoNode* node = nullptr;
        bool has_children = false;
    };

    // Depth-first list of currently visible rows (respecting `expanded`).
    std::vector<Row> flatten() const;

    // Navigation / interaction over the flattened view.
    std::size_t selected() const { return sel_; }
    void move(int delta);                  // j/k
    void toggle_selected();                // space: expand/collapse
    int selected_layer() const;            // layer of selection, or -1

private:
    void flatten_into(const TopoNode& n, int depth, std::vector<Row>& out) const;
    TopoNode* ensure_layer(int layer);

    TopoNode root_;
    int n_layers_ = 0;
    std::map<int, std::size_t> layer_index_;  // layer -> child index under "layers"
    std::size_t layers_group_ = 0;            // index of the "layers" group in root
    std::size_t sel_ = 0;
};

} // namespace llmtrace
