// src/progress_bar.hpp
#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <algorithm> // For std::min

class ProgressBar {
public:
    ProgressBar(long long total_steps, std::string description = "")
        : total_steps_(total_steps),
          description_(description),
          bar_width_(50), // Set a standard width for the bar
          start_time_(std::chrono::steady_clock::now())
    {
        // Immediately draw the initial, empty bar
        update(0);
    }

    void update(long long current_step) {
        float progress = 0.0f;
        if (total_steps_ > 0) {
            long long display_step = std::min(current_step, total_steps_);
            progress = static_cast<float>(display_step) / total_steps_;
        } else if (current_step > 0) {
            progress = 1.0; // If total is 0, any progress is 100%
        }
        
        int current_pos = static_cast<int>(bar_width_ * progress);

        // Optimization: only redraw if the bar's visual changes
        if (current_pos == last_pos_ && current_step > 0 && current_step < total_steps_) {
            return;
        }
        last_pos_ = current_pos;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        
        std::string eta_str;
        if (progress > 0.001 && current_step < total_steps_) {
            // Calculate ETA based on elapsed time and progress
            long long total_ms_estimated = static_cast<long long>(elapsed_ms / progress);
            long long remaining_ms = total_ms_estimated - elapsed_ms;
            long long eta_sec = remaining_ms / 1000;
            long long eta_min = eta_sec / 60;
            eta_sec %= 60;
            std::stringstream ss;
            ss << " | ETA: " << std::setfill('0') << std::setw(2) << eta_min << ":" << std::setw(2) << eta_sec;
            eta_str = ss.str();
        } else if (current_step >= total_steps_) {
            // Show total elapsed time at the end
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << elapsed_ms / 1000.0;
            eta_str = " | Elapsed: " + ss.str() + "s";
        } else {
            eta_str = " | ETA: ...";
        }

        // Use carriage return '\r' to update the line in-place
        std::cout << "\r" << description_ << " [";
        for (int i = 0; i < bar_width_; ++i) {
            if (i < current_pos) std::cout << "=";
            else if (i == current_pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0 << "%"
                  << " (" << current_step << "/" << total_steps_ << ")"
                  << eta_str << "  "; // Extra spaces to clear previous, longer lines
                  
        std::cout.flush();
    }
    
    void done() {
        update(total_steps_);
        std::cout << std::endl; // Finish with a newline
    }

private:
    long long total_steps_;
    std::string description_;
    int bar_width_;
    int last_pos_ = -1;
    std::chrono::steady_clock::time_point start_time_;
};

#endif // PROGRESS_BAR_HPP