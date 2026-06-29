#include "llmtrace/trace_io.hpp"

#include <sstream>

namespace llmtrace {

namespace {

constexpr const char* kMagic = "LLMTRACEv1";

// Tabs/newlines would corrupt the TSV layout; tensor names never legitimately
// contain them, so collapsing to spaces is a safe, lossless-enough sanitisation.
std::string sanitize(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    }
    return out;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

} // namespace

TraceWriter::TraceWriter(const std::string& path) : os_(path, std::ios::out | std::ios::trunc) {
    if (os_) os_ << kMagic << '\n';
}

void TraceWriter::write(const LayerEvent& e) {
    if (!os_) return;
    os_ << e.seq << '\t'
        << e.t_ms << '\t'
        << e.latency_ms << '\t'
        << e.layer << '\t'
        << static_cast<int>(e.op_class) << '\t'
        << sanitize(e.op) << '\t'
        << sanitize(e.name) << '\t'
        << e.shape[0] << '\t' << e.shape[1] << '\t' << e.shape[2] << '\t' << e.shape[3] << '\t'
        << e.n_dims << '\t'
        << sanitize(e.dtype) << '\t'
        << sanitize(e.device) << '\t'
        << (e.has_stats ? 1 : 0) << '\t'
        << e.mean << '\t' << e.vmin << '\t' << e.vmax << '\t' << e.sparsity << '\t'
        << (e.anomaly ? 1 : 0) << '\t'
        << sanitize(e.anomaly_msg) << '\n';
}

void TraceWriter::close() {
    if (os_) os_.flush();
    os_.close();
}

TraceReader::TraceReader(const std::string& path) : is_(path, std::ios::in) {
    ok_ = static_cast<bool>(is_);
}

bool TraceReader::read_all(std::vector<LayerEvent>& out) {
    if (!ok_) return false;
    std::string line;
    if (!std::getline(is_, line)) return false;
    if (line.rfind(kMagic, 0) != 0) return false;  // bad/missing magic

    while (std::getline(is_, line)) {
        if (line.empty()) continue;
        auto f = split_tabs(line);
        if (f.size() < 21) continue;  // skip malformed rows defensively
        LayerEvent e;
        try {
            e.seq = std::stoull(f[0]);
            e.t_ms = std::stod(f[1]);
            e.latency_ms = std::stod(f[2]);
            e.layer = std::stoi(f[3]);
            e.op_class = static_cast<OpClass>(std::stoi(f[4]));
            e.op = f[5];
            e.name = f[6];
            e.shape = {std::stoll(f[7]), std::stoll(f[8]), std::stoll(f[9]), std::stoll(f[10])};
            e.n_dims = std::stoi(f[11]);
            e.dtype = f[12];
            e.device = f[13];
            e.has_stats = f[14] == "1";
            e.mean = std::stof(f[15]);
            e.vmin = std::stof(f[16]);
            e.vmax = std::stof(f[17]);
            e.sparsity = std::stof(f[18]);
            e.anomaly = f[19] == "1";
            e.anomaly_msg = f[20];
        } catch (...) {
            continue;  // tolerate a corrupt row rather than aborting the whole trace
        }
        out.push_back(std::move(e));
    }
    return true;
}

} // namespace llmtrace
