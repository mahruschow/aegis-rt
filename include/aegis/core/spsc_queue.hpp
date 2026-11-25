#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace aegis::core {

// Hardware destructive interference size (64 bytes on x86_64 and ARMv8/Apple Silicon M-series uses 128B L2 lines).
// Using 128 bytes eliminates false sharing across both x86_64 and Apple Silicon / NVIDIA Grace/Orin ARM64 cores.
inline constexpr std::size_t kCacheLineBytes = 128;

/**
 * @brief Lock-free, bounded, zero-allocation Single-Producer Single-Consumer (SPSC) ring buffer.
 *
 * Designed for deterministic real-time sensor -> perception -> control pipelines:
 *  - Wait-free O(1) push and pop (no mutexes, condition variables, or syscalls).
 *  - Cache-line isolated producer/consumer indices + local shadow index caching
 *    to prevent MESI cache-coherence invalidation traffic when the queue is neither full nor empty.
 *  - Power-of-two capacity enables fast bitwise masking instead of integer modulo.
 */
template <typename T, std::size_t Capacity>
class SpscQueue {
    static_assert(Capacity >= 2 && ((Capacity & (Capacity - 1)) == 0),
                  "SpscQueue Capacity must be a power of 2 and >= 2.");
    static_assert(std::is_default_constructible_v<T>,
                  "SpscQueue element type T must be default constructible for static pre-allocation.");

public:
    SpscQueue() = default;
    ~SpscQueue() = default;

    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    [[nodiscard]] bool try_push(const T& item) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        return emplace_impl(item);
    }

    [[nodiscard]] bool try_push(T&& item) noexcept(std::is_nothrow_move_assignable_v<T>) {
        return emplace_impl(std::move(item));
    }

    [[nodiscard]] bool try_pop(T& out_item) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const std::size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == cached_tail_) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if (current_head == cached_tail_) {
                return false;  // Queue is empty
            }
        }

        out_item = std::move(buffer_[current_head & kIndexMask]);
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Push newest frame, dropping the oldest unread item if full (real-time sensor semantics).
     * Note: Only safe when called in single-threaded mode; for concurrent producer/consumer, prefer
     * try_push() on the producer or draining on the consumer side to preserve SPSC ownership.
     */
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t t = tail_.load(std::memory_order_acquire);
        const std::size_t h = head_.load(std::memory_order_acquire);
        return t - h;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_approx() == 0;
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity;
    }

private:
    static constexpr std::size_t kIndexMask = Capacity - 1;

    template <typename U>
    [[nodiscard]] bool emplace_impl(U&& item) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail - cached_head_ == Capacity) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (current_tail - cached_head_ == Capacity) {
                return false;  // Queue is full
            }
        }

        buffer_[current_tail & kIndexMask] = std::forward<U>(item);
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Producer-owned cache line
    alignas(kCacheLineBytes) std::atomic<std::size_t> tail_{0};
    std::size_t cached_head_{0};

    // Consumer-owned cache line
    alignas(kCacheLineBytes) std::atomic<std::size_t> head_{0};
    std::size_t cached_tail_{0};

    // Data storage isolated on its own cache line
    alignas(kCacheLineBytes) std::array<T, Capacity> buffer_{};
};

}  // namespace aegis::core
