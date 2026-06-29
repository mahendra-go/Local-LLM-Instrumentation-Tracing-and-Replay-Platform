#include "llmtrace/dashboard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>

namespace llmtrace {

using namespace ftxui;

namespace {

enum Focus { F_TOPO = 0, F_STREAM, F_ATTN, F_METRICS, F_ANOM, F_COUNT };

Color class_color(OpClass c) {
    switch (c) {
        case OpClass::Embedding: return Color::Cyan;
        case OpClass::Attention: return Color::Yellow;
        case OpClass::Mlp:       return Color::Green;
        case OpClass::Norm:      return Color::BlueLight;
        case OpClass::Output:    return Color::Magenta;
        case OpClass::Rope:      return Color::GrayLight;
        default:                 return Color::GrayDark;
    }
}

std::string fmt_time(double t_ms) {
    double s = t_ms / 1000.0;
    int mins = static_cast<int>(s / 60.0);
    double rem = s - mins * 60.0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%06.3f", mins, rem);
    return buf;
}

std::string fmt_shape(const LayerEvent& e) {
    std::string s = "[";
    for (int i = 0; i < e.n_dims && i < 4; ++i) {
        if (i) s += ",";
        s += std::to_string(e.shape[i]);
    }
    s += "]";
    return s;
}

Color heat_color(float v) {
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    int r, g, b;
    if (v < 0.5f) {           // blue -> green
        float t = v * 2.0f;
        r = 0; g = static_cast<int>(t * 255); b = static_cast<int>((1 - t) * 200 + 40);
    } else {                  // green -> red
        float t = (v - 0.5f) * 2.0f;
        r = static_cast<int>(t * 255); g = static_cast<int>((1 - t) * 255); b = 0;
    }
    return Color::RGB(static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b));
}

Element panel(const std::string& title, Element content, bool focused) {
    auto w = window(text(focused ? "▌" + title + " ◂" : " " + title), std::move(content));
    return focused ? (w | color(Color::GreenLight)) : (w | color(Color::GrayDark));
}

// ---- Panel 1: model topology tree ----
Element render_topology(SessionState& st, bool focused) {
    auto rows = st.topo_rows();
    int active = st.active_layer();
    Elements out;
    for (const auto& r : rows) {
        std::string indent(static_cast<std::size_t>(r.depth) * 2, ' ');
        std::string marker = r.has_children ? (r.expanded ? "▼ " : "▶ ") : "● ";
        std::string label = r.label;
        Element line = hbox({
            text(indent + marker),
            text(label) | color(class_color(r.op_class)),
            (r.is_layer && r.layer == active) ? text("  [active]") | color(Color::RedLight) | bold
                                              : text(""),
        });
        if (r.selected) line = line | bgcolor(Color::GrayDark) | bold;
        out.push_back(line);
    }
    if (out.empty()) out.push_back(text("(waiting for events…)") | dim);
    return vbox(std::move(out));
}

// ---- Panel 2: live event stream ----
Element render_stream(SessionState& st) {
    Elements out;
    out.push_back(hbox({
        text("  ID ") | bold, text("│ "), text("TIME      ") | bold, text("│ "),
        text("LAYER ") | bold, text("│ "), text("TYPE      ") | bold, text("│ "),
        text("DEVICE        ") | bold, text("│ "), text("TENSOR") | bold,
    }) | color(Color::GrayLight));
    out.push_back(separator());
    for (const auto& e : st.recent_events(24)) {
        char id[8]; std::snprintf(id, sizeof(id), "%5llu", (unsigned long long)e.seq);
        char ly[8]; std::snprintf(ly, sizeof(ly), "%4d", e.layer);
        out.push_back(hbox({
            text(std::string(id) + " "), text("│ "),
            text(fmt_time(e.t_ms) + " "), text("│ "),
            text(std::string(ly) + "  "), text("│ "),
            text(std::string(to_string(e.op_class)) + std::string(10 - std::min<size_t>(10, strlen(to_string(e.op_class))), ' '))
                | color(class_color(e.op_class)),
            text("│ "),
            text(e.device + std::string(14 - std::min<size_t>(14, e.device.size()), ' ')),
            text("│ "),
            text(e.name) | dim,
        }));
    }
    return vbox(std::move(out));
}

// ---- Panel 3: attention matrix heatmap ----
Element render_attention(SessionState& st, float contrast, int preferred_layer) {
    int layer = -1;
    auto m = st.attention(preferred_layer, layer);
    if (m.empty()) return vbox({text("(no attention captured yet)") | dim});
    Elements rows;
    rows.push_back(text("layer " + std::to_string(layer) + " · query↓ key→ · causal") | color(Color::Yellow));
    for (std::size_t i = 0; i < m.size(); ++i) {
        Elements cells;
        char idx[8]; std::snprintf(idx, sizeof(idx), "%2zu ", i);
        cells.push_back(text(idx) | dim);
        for (std::size_t j = 0; j < m[i].size(); ++j) {
            float v = std::pow(m[i][j], contrast);
            const char* ch = v < 0.04f ? "  " : v < 0.2f ? "░░" : v < 0.45f ? "▒▒"
                            : v < 0.7f ? "▓▓" : "██";
            cells.push_back(text(ch) | color(heat_color(v)));
        }
        rows.push_back(hbox(std::move(cells)));
    }
    rows.push_back(text("low ░▒▓█ high  [+/-] contrast=" +
                        std::to_string(contrast).substr(0, 3)) | dim);
    return vbox(std::move(rows));
}

// ---- Panel 4: runtime metrics inspector ----
Element render_metrics(SessionState& st, int layer) {
    LayerEvent e;
    if (layer < 0 || !st.last_event_for_layer(layer, e)) {
        layer = st.active_layer();
        if (layer < 0 || !st.last_event_for_layer(layer, e)) {
            return vbox({text("(select a layer)") | dim});
        }
    }
    auto kv = [](const std::string& k, Element v) {
        return hbox({text(k) | color(Color::GrayLight) | size(WIDTH, EQUAL, 14), v});
    };
    Element sparsity = hbox({gauge(e.sparsity) | size(WIDTH, EQUAL, 16) | color(Color::Cyan),
                             text(" " + std::to_string(e.sparsity * 100.0f).substr(0, 4) + "%")});
    return vbox({
        kv("Layer / op", text("L" + std::to_string(e.layer) + "  " + e.name)
                             | color(class_color(e.op_class))),
        kv("ggml op", text(e.op)),
        kv("Tensor shape", text(fmt_shape(e))),
        kv("Dtype", text(e.dtype)),
        kv("Device", text(e.device) | (e.device.find("CPU") != std::string::npos
                                           ? color(Color::RedLight) : color(Color::Default))),
        kv("Latency", text(std::to_string(e.latency_ms).substr(0, 5) + " ms")),
        kv("Sparsity", sparsity),
        kv("mean/min/max", text(std::to_string(e.mean).substr(0, 5) + " / " +
                                std::to_string(e.vmin).substr(0, 5) + " / " +
                                std::to_string(e.vmax).substr(0, 5))),
    });
}

// ---- Panel 5: numerical anomaly ledger ----
Element render_anomalies(SessionState& st) {
    Elements out;
    auto rows = st.recent_anomalies(6);
    if (rows.empty()) return vbox({text("no anomalies — all within normal bounds ✓") | color(Color::Green)});
    for (const auto& e : rows) {
        bool fallback = e.anomaly_msg.find("fallback") != std::string::npos;
        out.push_back(hbox({
            text(fmt_time(e.t_ms) + " "),
            text(fallback ? "✖ " : "⚠ ") | color(fallback ? Color::Red : Color::Yellow),
            text(e.anomaly_msg) | color(fallback ? Color::RedLight : Color::YellowLight),
        }));
    }
    return vbox(std::move(out));
}

Element build_frame(SessionState& st, int focus, float contrast, int preferred_layer) {
    char stats[128];
    std::snprintf(stats, sizeof(stats), "events %llu · layers %d · %.0f ev/s · %.1f ms total",
                  (unsigned long long)st.total_events(), st.n_layers(),
                  st.events_per_sec(), st.total_latency_ms());
    Element header = hbox({
        text(" llmtrace ") | bold | bgcolor(Color::Green) | color(Color::Black),
        text(" [Tab] focus  [j/k] nav  [Space] expand  [+/-] contrast  [q] quit ") | dim,
        filler(),
        text(std::string(stats) + " ") | color(Color::GrayLight),
    });

    Element top = hbox({
        panel("1 MODEL TOPOLOGY", render_topology(st, focus == F_TOPO), focus == F_TOPO)
            | size(WIDTH, EQUAL, 34),
        panel("2 LIVE EVENT STREAM", render_stream(st), focus == F_STREAM) | flex,
    }) | flex;

    Element mid = hbox({
        panel("3 ATTENTION MATRIX", render_attention(st, contrast, preferred_layer), focus == F_ATTN) | flex,
        panel("4 RUNTIME METRICS", render_metrics(st, preferred_layer), focus == F_METRICS)
            | size(WIDTH, EQUAL, 44),
    }) | flex;

    Element bottom = panel("5 NUMERICAL ANOMALY LEDGER", render_anomalies(st), focus == F_ANOM)
                     | size(HEIGHT, EQUAL, 9);

    return vbox({header, top, mid, bottom});
}

} // namespace

void run_dashboard(DataSource& src, const DashboardOptions& opt) {
    SessionState st(opt.event_cap, opt.anomaly_cap, opt.seq_len);
    auto screen = ScreenInteractive::Fullscreen();

    std::atomic<bool> running{true};
    int focus = F_TOPO;
    float contrast = 1.0f;

    // Producer: pull events and pace them like a live run.
    std::thread producer([&] {
        LayerEvent e;
        while (running.load()) {
            if (!src.next(e)) break;
            st.ingest(e);
            double ms = src.pacing_ms() / (opt.speed > 0 ? opt.speed : 1.0);
            if (ms < 1.0) ms = 1.0;
            if (ms > 40.0) ms = 40.0;
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
            screen.PostEvent(Event::Custom);
        }
    });

    auto root = Renderer([&] {
        int sel = st.topo_selected_layer();
        return build_frame(st, focus, contrast, sel);
    });

    root |= CatchEvent([&](const Event& ev) {
        if (ev == Event::Character("q") || ev == Event::Character("Q") || ev == Event::Escape) {
            running.store(false);
            screen.Exit();
            return true;
        }
        if (ev == Event::Tab) { focus = (focus + 1) % F_COUNT; return true; }
        if (ev == Event::TabReverse) { focus = (focus + F_COUNT - 1) % F_COUNT; return true; }
        if (ev == Event::Character("j") || ev == Event::ArrowDown) { st.topo_move(1); return true; }
        if (ev == Event::Character("k") || ev == Event::ArrowUp) { st.topo_move(-1); return true; }
        if (ev == Event::Character(" ") || ev == Event::Return) { st.topo_toggle(); return true; }
        if (ev == Event::Character("+") || ev == Event::Character("=")) { contrast = std::max(0.2f, contrast - 0.2f); return true; }
        if (ev == Event::Character("-") || ev == Event::Character("_")) { contrast = std::min(4.0f, contrast + 0.2f); return true; }
        return false;
    });

    screen.Loop(root);
    running.store(false);
    if (producer.joinable()) producer.join();
}

std::string render_preview(DataSource& src, const DashboardOptions& opt,
                           int n_events, int width, int height) {
    SessionState st(opt.event_cap, opt.anomaly_cap, opt.seq_len);
    LayerEvent e;
    for (int i = 0; i < n_events && src.next(e); ++i) st.ingest(e);
    auto element = build_frame(st, F_TOPO, 1.0f, st.active_layer());
    auto screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Render(screen, element);
    return screen.ToString();
}

} // namespace llmtrace
