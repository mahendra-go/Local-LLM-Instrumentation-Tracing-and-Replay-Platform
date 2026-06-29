#include "llmtrace/event.hpp"

#include <cctype>

namespace llmtrace {

const char* to_string(OpClass c) {
    switch (c) {
        case OpClass::Embedding: return "Embedding";
        case OpClass::Attention: return "Attention";
        case OpClass::Mlp:       return "MLP";
        case OpClass::Norm:      return "Norm";
        case OpClass::Output:    return "Output";
        case OpClass::Rope:      return "RoPE";
        case OpClass::Other:     return "Other";
    }
    return "Other";
}

namespace {

bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

// Pulls a transformer block index out of a tensor name. Handles both ggml runtime
// names ("ffn_out-15", "kq-3", "l_out-7") and gguf weight names ("blk.15.attn_q").
int parse_layer(const std::string& name) {
    auto pos = name.find("blk.");
    if (pos != std::string::npos) {
        pos += 4;
        int v = 0;
        bool any = false;
        while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
            v = v * 10 + (name[pos] - '0');
            ++pos;
            any = true;
        }
        if (any) return v;
    }
    // trailing "-<int>"
    auto dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size()) {
        int v = 0;
        bool any = false;
        for (std::size_t i = dash + 1; i < name.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(name[i]))) { any = false; break; }
            v = v * 10 + (name[i] - '0');
            any = true;
        }
        if (any) return v;
    }
    return -1;
}

} // namespace

OpClass classify(const std::string& name, int* out_layer) {
    std::string n;
    n.reserve(name.size());
    for (char ch : name) n.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

    if (out_layer) *out_layer = parse_layer(n);

    if (contains(n, "embd") || contains(n, "embed") || contains(n, "token")) return OpClass::Embedding;
    if (contains(n, "rope") || contains(n, "rot")) return OpClass::Rope;
    if (contains(n, "norm") || contains(n, "rms") || contains(n, "ln")) return OpClass::Norm;
    if (contains(n, "attn") || contains(n, "kq") || contains(n, "kqv") ||
        contains(n, "qcur") || contains(n, "kcur") || contains(n, "vcur") ||
        contains(n, "soft_max") || contains(n, "softmax")) return OpClass::Attention;
    if (contains(n, "ffn") || contains(n, "mlp") || contains(n, "gate") ||
        contains(n, "up") || contains(n, "down")) return OpClass::Mlp;
    if (contains(n, "output") || contains(n, "result") || contains(n, "logits")) return OpClass::Output;
    return OpClass::Other;
}

} // namespace llmtrace
