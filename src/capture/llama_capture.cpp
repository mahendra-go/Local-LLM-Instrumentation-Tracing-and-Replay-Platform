// llama.cpp capture backend.
//
// Non-invasive instrumentation: we register a ggml eval callback on the inference
// context. ggml invokes it for every node of the compute graph as the graph runs,
// so we observe each tensor's name/op/shape/dtype/timing — and, for host-resident
// F32 tensors, real activation statistics — without modifying the model or its code.
#include "llmtrace/llama_capture.hpp"

#ifndef LLMTRACE_ENABLE_LLAMA

namespace llmtrace {
struct LlamaCaptureSource::Impl {};
LlamaCaptureSource::LlamaCaptureSource(CaptureConfig) : impl_(nullptr) {}
LlamaCaptureSource::~LlamaCaptureSource() = default;
bool LlamaCaptureSource::next(LayerEvent&) { return false; }
bool LlamaCaptureSource::ok() const { return false; }
std::string LlamaCaptureSource::error() const { return "built without LLMTRACE_ENABLE_LLAMA"; }
} // namespace llmtrace

#else

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace llmtrace {

namespace {
void silent_log(enum ggml_log_level, const char*, void*) {}  // keep llama logs off the TUI
}

struct LlamaCaptureSource::Impl {
    CaptureConfig cfg;

    std::mutex m;
    std::condition_variable cv;
    std::deque<LayerEvent> q;
    std::atomic<bool> done{false};
    std::atomic<bool> stop{false};
    bool ok = true;
    std::string err;

    // touched only by the inference thread
    std::uint64_t seq = 0;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last;

    std::thread worker;

    explicit Impl(CaptureConfig c) : cfg(std::move(c)) {
        worker = std::thread([this] { run(); });
    }

    ~Impl() {
        stop.store(true);
        if (worker.joinable()) worker.join();
    }

    void push(LayerEvent&& e) {
        {
            std::lock_guard<std::mutex> lk(m);
            q.push_back(std::move(e));
        }
        cv.notify_one();
    }

    void finish(const std::string& error = "") {
        if (!error.empty()) { ok = false; err = error; }
        done.store(true);
        cv.notify_all();
    }

    // Called by ggml for every graph node. `ask==true`: return whether we want the
    // node's data after it computes. `ask==false`: the node has executed.
    bool on_node(struct ggml_tensor* t) {
        auto now = std::chrono::steady_clock::now();
        double t_ms = std::chrono::duration<double, std::milli>(now - start).count();
        double lat = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;

        LayerEvent e;
        e.seq = seq++;
        e.t_ms = t_ms;
        e.latency_ms = lat;
        e.name = ggml_get_name(t);
        e.op = ggml_op_name(t->op);
        e.op_class = classify(e.name, &e.layer);

        int nd = 1;
        for (int i = GGML_MAX_DIMS - 1; i > 0; --i) {
            if (t->ne[i] > 1) { nd = i + 1; break; }
        }
        e.n_dims = nd;
        for (int i = 0; i < GGML_MAX_DIMS && i < 4; ++i) e.shape[i] = t->ne[i];
        e.dtype = ggml_type_name(t->type);
        e.device = t->data ? "CPU" : "Metal/GPU";

        // Real activation stats for host-resident, contiguous F32 tensors.
        if (t->type == GGML_TYPE_F32 && t->data && ggml_is_contiguous(t)) {
            std::int64_t n = ggml_nelements(t);
            std::int64_t cap = n < 8192 ? n : 8192;
            const float* p = static_cast<const float*>(t->data);
            double sum = 0.0;
            float mn = 1e30f, mx = -1e30f;
            std::int64_t zeros = 0, bad = 0;
            for (std::int64_t i = 0; i < cap; ++i) {
                float v = p[i];
                if (std::isnan(v) || std::isinf(v)) { ++bad; continue; }
                sum += v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                if (std::fabs(v) < 1e-6f) ++zeros;
            }
            if (cap > 0) {
                e.has_stats = true;
                e.mean = static_cast<float>(sum / static_cast<double>(cap));
                e.vmin = mn;
                e.vmax = mx;
                e.sparsity = static_cast<float>(zeros) / static_cast<float>(cap);
                float amax = std::fabs(mn) > std::fabs(mx) ? std::fabs(mn) : std::fabs(mx);
                if (bad > 0) {
                    e.anomaly = true;
                    e.anomaly_msg = "NaN/Inf in " + e.name + " (L" + std::to_string(e.layer) + ")";
                } else if (amax > 30.0f) {
                    e.anomaly = true;
                    e.anomaly_msg = "Outlier feature L" + std::to_string(e.layer) +
                                    ": |val| " + std::to_string(amax).substr(0, 5) + " > 30 (" +
                                    e.name + ")";
                }
            }
        }
        push(std::move(e));
        return true;  // IMPORTANT: returning false here tells ggml to abort the graph
    }

    void run() {
        llama_log_set(silent_log, nullptr);
        ggml_log_set(silent_log, nullptr);
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = cfg.n_gpu_layers;  // 0 => CPU, so F32 activations are host-readable
        llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
        if (!model) { finish("failed to load model: " + cfg.model_path); llama_backend_free(); return; }

        const llama_vocab* vocab = llama_model_get_vocab(model);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = cfg.n_ctx;
        cp.cb_eval = &Impl::eval_trampoline;
        cp.cb_eval_user_data = this;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) { finish("failed to create context"); llama_model_free(model); llama_backend_free(); return; }

        // Tokenize the prompt.
        std::vector<llama_token> toks(cfg.prompt.size() + 8);
        int n = llama_tokenize(vocab, cfg.prompt.c_str(), (int)cfg.prompt.size(),
                               toks.data(), (int)toks.size(), /*add_special*/true, /*parse_special*/true);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, cfg.prompt.c_str(), (int)cfg.prompt.size(),
                                                         toks.data(), (int)toks.size(), true, true); }
        if (n <= 0) { finish("tokenization failed"); llama_free(ctx); llama_model_free(model); llama_backend_free(); return; }
        toks.resize(n);

        start = std::chrono::steady_clock::now();
        last = start;

        // Prefill: one decode over the whole prompt (fires the callback for every node).
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int)toks.size())) != 0) {
            finish("decode (prefill) failed");
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return;
        }

        // Greedy generation; each step is a fresh forward pass we capture.
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int g = 0; g < cfg.n_predict && !stop.load(); ++g) {
            const float* logits = llama_get_logits_ith(ctx, -1);
            if (!logits) break;
            int best = 0;
            float bestv = logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > bestv) { bestv = logits[i]; best = i; }
            }
            llama_token tok = (llama_token)best;
            if (llama_vocab_is_eog(vocab, tok)) break;
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) break;
        }

        finish();
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }

    static bool eval_trampoline(struct ggml_tensor* t, bool ask, void* user_data) {
        auto* self = static_cast<Impl*>(user_data);
        if (ask) return true;            // yes, call us back once the node has data
        return self->on_node(t);
    }
};

LlamaCaptureSource::LlamaCaptureSource(CaptureConfig cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg))) {}

LlamaCaptureSource::~LlamaCaptureSource() = default;

bool LlamaCaptureSource::next(LayerEvent& out) {
    std::unique_lock<std::mutex> lk(impl_->m);
    impl_->cv.wait(lk, [&] { return !impl_->q.empty() || impl_->done.load(); });
    if (!impl_->q.empty()) {
        out = std::move(impl_->q.front());
        impl_->q.pop_front();
        return true;
    }
    return false;  // done and drained
}

bool LlamaCaptureSource::ok() const { return impl_->ok; }
std::string LlamaCaptureSource::error() const { return impl_->err; }

} // namespace llmtrace

#endif // LLMTRACE_ENABLE_LLAMA
