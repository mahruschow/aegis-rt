#include "aegis/core/spsc_queue.hpp"
#include "aegis/core/telemetry.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>

int main() {
    using aegis::core::SpscQueue;
    SpscQueue<uint64_t, 1024> queue;

    // 1. Single-threaded boundary & wrap-around checks
    assert(queue.empty());
    for (uint64_t i = 0; i < 1024; ++i) {
        assert(queue.try_push(i));
    }
    assert(!queue.try_push(9999));  // Full check

    for (uint64_t i = 0; i < 512; ++i) {
        uint64_t val = 0;
        assert(queue.try_pop(val));
        assert(val == i);
    }
    // Wrap around ring indices
    for (uint64_t i = 1024; i < 1536; ++i) {
        assert(queue.try_push(i));
    }
    for (uint64_t i = 512; i < 1536; ++i) {
        uint64_t val = 0;
        assert(queue.try_pop(val));
        assert(val == i);
    }
    assert(queue.empty());

    // 2. Concurrent Producer-Consumer stress & latency benchmark (500,000 messages)
    constexpr uint64_t kNumMessages = 500'000;
    const uint64_t t0 = aegis::core::now_ns();

    std::thread producer([&]() {
        for (uint64_t i = 1; i <= kNumMessages; ++i) {
            while (!queue.try_push(i)) {
                // Spin-wait as in pinned real-time threads
            }
        }
    });

    uint64_t expected = 1;
    std::thread consumer([&]() {
        uint64_t val = 0;
        while (expected <= kNumMessages) {
            if (queue.try_pop(val)) {
                assert(val == expected);
                ++expected;
            }
        }
    });

    producer.join();
    consumer.join();
    const uint64_t t1 = aegis::core::now_ns();
    const double ns_per_handoff = static_cast<double>(t1 - t0) / static_cast<double>(kNumMessages);

    std::cout << "[PASS] test_spsc_queue: 500,000 lock-free handoffs verified in-order ("
              << ns_per_handoff << " ns/msg).\n";
    return 0;
}
