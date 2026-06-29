#include <cstdio>

#ifndef LLMTRACE_VERSION
#define LLMTRACE_VERSION "0.0.0"
#endif

int main(int /*argc*/, char** /*argv*/) {
    std::printf("llmtrace %s\n", LLMTRACE_VERSION);
    std::printf("Local LLM Instrumentation, Tracing & Replay Platform\n");
    std::printf("(scaffold) build OK\n");
    return 0;
}
