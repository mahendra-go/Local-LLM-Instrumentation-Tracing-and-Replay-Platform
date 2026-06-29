#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace llmtrace {

// Fixed-capacity, thread-safe ring buffer. When full, pushing overwrites the oldest
// element so memory usage stays flat regardless of how long capture runs — important
// for long inference sessions (the project explicitly calls this out).
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : cap_(capacity ? capacity : 1), buf_(cap_) {}

    void push(const T& value) {
        std::lock_guard<std::mutex> lk(m_);
        buf_[head_] = value;
        head_ = (head_ + 1) % cap_;
        if (size_ < cap_) {
            ++size_;
        } else {
            tail_ = (tail_ + 1) % cap_;  // overwrite oldest
        }
        ++total_;
    }

    // Oldest-first copy of the currently retained elements.
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<T> out;
        out.reserve(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            out.push_back(buf_[(tail_ + i) % cap_]);
        }
        return out;
    }

    // Newest-first copy of up to `n` most recent elements.
    std::vector<T> recent(std::size_t n) const {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<T> out;
        std::size_t k = n < size_ ? n : size_;
        out.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            std::size_t idx = (head_ + cap_ - 1 - i) % cap_;
            out.push_back(buf_[idx]);
        }
        return out;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return size_;
    }
    std::size_t capacity() const { return cap_; }
    std::uint64_t total() const {
        std::lock_guard<std::mutex> lk(m_);
        return total_;
    }

private:
    std::size_t cap_;
    std::vector<T> buf_;
    std::size_t head_ = 0;   // next write position
    std::size_t tail_ = 0;   // oldest element
    std::size_t size_ = 0;
    std::uint64_t total_ = 0;
    mutable std::mutex m_;
};

} // namespace llmtrace
