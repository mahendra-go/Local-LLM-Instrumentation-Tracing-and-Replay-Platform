#include <cstdio>
#include <cstring>
#include <string>

#include "llmtrace/event.hpp"
#include "llmtrace/ring_buffer.hpp"
#include "llmtrace/topology.hpp"

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
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) return run_selftest();
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }
    print_usage();
    return 0;
}
