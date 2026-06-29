#pragma once
#include <string>

#include "llmtrace/session.hpp"
#include "llmtrace/source.hpp"

namespace llmtrace {

struct DashboardOptions {
    int seq_len = 16;       // attention window size for the heatmap
    double speed = 1.0;     // playback speed multiplier for live pacing
    std::size_t event_cap = 4096;
    std::size_t anomaly_cap = 256;
};

// Runs the interactive FTXUI dashboard, streaming events from `src` on a background
// thread until the user quits (q / Esc). Blocks until exit.
void run_dashboard(DataSource& src, const DashboardOptions& opt);

// Renders a single dashboard frame to a string (no interactivity). Pulls `n_events`
// from `src` first. Used to preview/verify the layout in a non-TTY environment.
std::string render_preview(DataSource& src, const DashboardOptions& opt,
                           int n_events, int width, int height);

} // namespace llmtrace
