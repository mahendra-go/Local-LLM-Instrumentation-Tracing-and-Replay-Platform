#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "llmtrace/event.hpp"

namespace llmtrace {

// Append-only writer for the line-oriented .trace format:
//   line 1: "LLMTRACEv1"
//   line N: tab-separated LayerEvent fields (see trace_io.cpp for the schema)
// The format is intentionally human-inspectable so a trace can be diffed / grepped.
class TraceWriter {
public:
    explicit TraceWriter(const std::string& path);
    bool ok() const { return static_cast<bool>(os_); }
    void write(const LayerEvent& e);
    void close();

private:
    std::ofstream os_;
};

// Loads a whole .trace file into memory. Traces are bounded (ring-buffer sized
// captures), so eager loading keeps replay logic simple.
class TraceReader {
public:
    explicit TraceReader(const std::string& path);
    bool ok() const { return ok_; }
    bool read_all(std::vector<LayerEvent>& out);

private:
    std::ifstream is_;
    bool ok_ = false;
};

} // namespace llmtrace
