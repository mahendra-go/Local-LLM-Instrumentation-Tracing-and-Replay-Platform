#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llmtrace/dashboard.hpp"
#include "llmtrace/event.hpp"
#ifdef LLMTRACE_ENABLE_LLAMA
#include "llmtrace/llama_capture.hpp"
#endif
#include "llmtrace/ring_buffer.hpp"
#include "llmtrace/session.hpp"
#include "llmtrace/source.hpp"
#include "llmtrace/topology.hpp"
#include "llmtrace/trace_io.hpp"

#ifndef LLMTRACE_VERSION
#define LLMTRACE_VERSION "0.0.0"
#endif

namespace {

int run_selftest() {
    using namespace llmtrace;

    // 1) classify
    struct Case { const char* name; OpClass cls; int layer; };
    const Case cases[] = {
        {"token_embd",   OpClass::Embedding, -1},
        {"attn_norm-0",  OpClass::Norm,       0},
        {"kq-15",        OpClass::Attention, 15},
        {"ffn_down-3",   OpClass::Mlp,        3},
        {"result_output",OpClass::Output,    -1},
    };
    int fails = 0;
    for (const auto& c : cases) {
        int layer = -2;
        OpClass got = classify(c.name, &layer);
        bool ok = (got == c.cls) && (layer == c.layer);
        std::printf("  classify %-14s -> %-10s layer=%-3d %s\n",
                    c.name, to_string(got), layer, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    }

    // 2) ring buffer overwrite semantics
    RingBuffer<int> rb(4);
    for (int i = 0; i < 10; ++i) rb.push(i);
    auto snap = rb.snapshot();
    bool rb_ok = snap.size() == 4 && snap.front() == 6 && snap.back() == 9 && rb.total() == 10;
    std::printf("  ringbuffer size=%zu total=%llu front=%d back=%d %s\n",
                snap.size(), (unsigned long long)rb.total(),
                snap.empty() ? -1 : snap.front(), snap.empty() ? -1 : snap.back(),
                rb_ok ? "ok" : "FAIL");
    if (!rb_ok) ++fails;

    // 3) topology builds layers from events
    Topology topo;
    for (int l = 0; l < 3; ++l) {
        for (const char* nm : {"attn_norm", "kq", "ffn_down"}) {
            LayerEvent e;
            e.name = std::string(nm) + "-" + std::to_string(l);
            e.op_class = classify(e.name, &e.layer);
            topo.observe(e);
        }
    }
    bool topo_ok = topo.n_layers() == 3 && topo.flatten().size() >= 5;
    std::printf("  topology n_layers=%d rows=%zu %s\n",
                topo.n_layers(), topo.flatten().size(), topo_ok ? "ok" : "FAIL");
    if (!topo_ok) ++fails;

    std::printf("selftest: %s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}

// Headless driver: pull N synthetic events through the pipeline, optionally record
// them, print a summary. Lets us validate capture + trace I/O without a terminal.
int run_headless_synthetic(int n, int layers, const char* record_path) {
    using namespace llmtrace;
    SyntheticSource src(SyntheticSource::Config{layers, /*tokens*/100, 32, 16,
                                                4096, 11008, /*loop*/false, 7});
    RingBuffer<LayerEvent> rb(4096);
    Topology topo;
    TraceWriter writer(record_path ? record_path : "");
    int anomalies = 0;
    LayerEvent e;
    int count = 0;
    for (; count < n && src.next(e); ++count) {
        rb.push(e);
        topo.observe(e);
        if (record_path) writer.write(e);
        if (e.anomaly) ++anomalies;
    }
    if (record_path) writer.close();
    std::printf("captured %d events | layers=%d | anomalies=%d | ring=%zu/%zu\n",
                count, topo.n_layers(), anomalies, rb.size(), rb.capacity());
    if (record_path) std::printf("recorded -> %s\n", record_path);
    return 0;
}

int run_headless_replay(const char* path) {
    using namespace llmtrace;
    std::vector<LayerEvent> events;
    TraceReader reader(path);
    if (!reader.ok() || !reader.read_all(events)) {
        std::fprintf(stderr, "error: cannot read trace '%s'\n", path);
        return 1;
    }
    Topology topo;
    int anomalies = 0;
    double total_lat = 0.0;
    for (const auto& e : events) {
        topo.observe(e);
        if (e.anomaly) ++anomalies;
        total_lat += e.latency_ms;
    }
    std::printf("replayed %zu events | layers=%d | anomalies=%d | total_latency=%.2f ms\n",
                events.size(), topo.n_layers(), anomalies, total_lat);
    if (!events.empty()) {
        const auto& f = events.front();
        std::printf("first: seq=%llu %s op=%s shape=[%lld,%lld] %s\n",
                    (unsigned long long)f.seq, f.name.c_str(), f.op.c_str(),
                    (long long)f.shape[0], (long long)f.shape[1], f.dtype.c_str());
    }
    return 0;
}

void print_usage() {
    std::printf("llmtrace %s — Local LLM Instrumentation, Tracing & Replay Platform\n\n",
                LLMTRACE_VERSION);
    std::printf("Usage:\n");
    std::printf("  llmtrace --selftest                 run internal self-checks\n");
    std::printf("  llmtrace --synthetic [--layers N]   live dashboard on synthetic data\n");
    std::printf("  llmtrace --replay FILE              replay a recorded .trace\n");
    std::printf("  llmtrace --model FILE --prompt P [--record FILE]\n");
    std::printf("                                      capture a real llama.cpp run\n");
}

} // namespace

int main(int argc, char** argv) {
    using namespace llmtrace;
    bool headless = false, synthetic = false, preview = false;
    int n = 200, layers = 32, seq_len = 16, n_predict = 24, n_gpu_layers = 0;
    double speed = 1.0;
    const char* record_path = nullptr;
    const char* replay_path = nullptr;
    const char* model_path = nullptr;
    const char* prompt = "The capital of France is";

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto val = [&](const char* def) -> const char* {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if (std::strcmp(a, "--selftest") == 0) return run_selftest();
        else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) { print_usage(); return 0; }
        else if (std::strcmp(a, "--headless") == 0) headless = true;
        else if (std::strcmp(a, "--preview") == 0) preview = true;
        else if (std::strcmp(a, "--synthetic") == 0) synthetic = true;
        else if (std::strcmp(a, "--layers") == 0) layers = std::atoi(val("32"));
        else if (std::strcmp(a, "--count") == 0) n = std::atoi(val("200"));
        else if (std::strcmp(a, "--seq") == 0) seq_len = std::atoi(val("16"));
        else if (std::strcmp(a, "--speed") == 0) speed = std::atof(val("1.0"));
        else if (std::strcmp(a, "--record") == 0) record_path = val(nullptr);
        else if (std::strcmp(a, "--replay") == 0) replay_path = val(nullptr);
        else if (std::strcmp(a, "--model") == 0) model_path = val(nullptr);
        else if (std::strcmp(a, "--prompt") == 0) prompt = val(prompt);
        else if (std::strcmp(a, "--predict") == 0) n_predict = std::atoi(val("24"));
        else if (std::strcmp(a, "--ngl") == 0) n_gpu_layers = std::atoi(val("0"));
    }

    if (replay_path && headless) return run_headless_replay(replay_path);
    if (synthetic && headless) return run_headless_synthetic(n, layers, record_path);

    DashboardOptions opt;
    opt.seq_len = seq_len;
    opt.speed = speed;

    // Build the chosen data source.
    if (replay_path) {
        std::vector<LayerEvent> events;
        TraceReader reader(replay_path);
        if (!reader.ok() || !reader.read_all(events)) {
            std::fprintf(stderr, "error: cannot read trace '%s'\n", replay_path);
            return 1;
        }
        ReplaySource src(std::move(events), /*loop*/true);
        if (preview) { std::printf("%s\n", render_preview(src, opt, n, 120, 44).c_str()); return 0; }
        run_dashboard(src, opt);
        return 0;
    }

    if (model_path) {
#ifdef LLMTRACE_ENABLE_LLAMA
        CaptureConfig cc;
        cc.model_path = model_path;
        cc.prompt = prompt;
        cc.n_predict = n_predict;
        cc.n_gpu_layers = n_gpu_layers;
        cc.seq_len = seq_len;
        LlamaCaptureSource src(cc);

        if (headless) {  // drain the whole run, optionally record a trace
            TraceWriter writer(record_path ? record_path : "");
            int count = 0, anomalies = 0;
            LayerEvent e;
            while (src.next(e)) {
                if (record_path) writer.write(e);
                if (e.anomaly) ++anomalies;
                ++count;
            }
            if (record_path) writer.close();
            if (!src.ok()) { std::fprintf(stderr, "capture error: %s\n", src.error().c_str()); return 1; }
            std::printf("captured %d events | anomalies=%d%s%s\n", count, anomalies,
                        record_path ? " | recorded -> " : "", record_path ? record_path : "");
            return 0;
        }
        run_dashboard(src, opt);
        if (!src.ok()) { std::fprintf(stderr, "capture error: %s\n", src.error().c_str()); return 1; }
        return 0;
#else
        std::fprintf(stderr, "error: rebuild with -DLLMTRACE_ENABLE_LLAMA=ON to capture a model.\n");
        return 1;
#endif
    }

    if (synthetic) {
        SyntheticSource::Config c;
        c.layers = layers;
        c.seq_len = seq_len;
        c.loop = true;
        SyntheticSource src(c);
        if (preview) { std::printf("%s\n", render_preview(src, opt, n, 120, 44).c_str()); return 0; }
        run_dashboard(src, opt);
        return 0;
    }

    print_usage();
    return 0;
}
