// C++20, Apple clang / libc++, macOS. See index.md for the experiment's scope.
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <thread>
#include <vector>

#ifndef STACK_COUNT_RETRIES
#define STACK_COUNT_RETRIES 1
#endif
static_assert(STACK_COUNT_RETRIES == 0 || STACK_COUNT_RETRIES == 1);

struct Node {
    Node* next = nullptr;
    std::uint64_t id = 0;
};

class MutexStack {
    std::mutex mutex_;
    Node* head_ = nullptr;

public:
    void push(Node* node, std::uint64_t&) {
        std::lock_guard<std::mutex> lock(mutex_);
        node->next = head_;
        head_ = node;
    }

    Node* pop(std::uint64_t&) {
        std::lock_guard<std::mutex> lock(mutex_);
        Node* node = head_;
        if (node) head_ = node->next;
        return node;
    }

    // Called only after joining every worker.
    bool empty() const { return head_ == nullptr; }
};

class TreiberStack {
    static_assert(std::atomic<Node*>::is_always_lock_free,
                  "This benchmark requires lock-free pointer atomics");
    std::atomic<Node*> head_{nullptr};

public:
    void push(Node* node, std::uint64_t& failures) {
        Node* expected = head_.load(std::memory_order_relaxed);
        do {
            // This node is private until the successful release CAS.
            node->next = expected;
            if (head_.compare_exchange_weak(expected, node,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) return;
            if constexpr (STACK_COUNT_RETRIES) ++failures;
        } while (true);
    }

    Node* pop(std::uint64_t& failures) {
        Node* expected = head_.load(std::memory_order_acquire);
        while (expected) {
            Node* next = expected->next;
            if (head_.compare_exchange_weak(expected, next,
                                           std::memory_order_acquire,
                                           std::memory_order_acquire)) return expected;
            // The failed CAS refreshes expected; acquire protects its next read.
            if constexpr (STACK_COUNT_RETRIES) ++failures;
        }
        return nullptr;
    }

    bool empty() const { return head_.load(std::memory_order_relaxed) == nullptr; }
};

// Separate each worker's final counter; the hot loop uses a local variable.
struct alignas(128) Stats { std::uint64_t failures = 0; };

struct Result {
    double wall_seconds;
    double cpu_seconds;
    std::uint64_t failures;
};

double cpu_seconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error("getrusage failed");
    return static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec)
         + static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
}

template<class Stack>
Result run(std::size_t pairs, unsigned threads) {
    Stack stack;
    // A node is pushed exactly once. Nothing is freed or reused until all joins.
    // In particular, popped nodes' next fields remain immutable for stale readers.
    auto nodes = std::make_unique<Node[]>(pairs);
    std::vector<Node*> popped(pairs, nullptr);
    std::vector<Stats> stats(threads);
    for (std::size_t i = 0; i < pairs; ++i) nodes[i].id = i;

    std::latch ready(threads), done(threads);
    std::atomic<bool> go{false};
    bool cancelled = false; // Published by go if thread creation fails.
    std::vector<std::thread> workers;
    workers.reserve(threads);
    // Wake and join existing threads if a later thread creation fails.
    try {
        for (unsigned t = 0; t < threads; ++t) {
            const auto begin = pairs * t / threads;
            const auto end = pairs * (t + 1) / threads;
            workers.emplace_back([&, t, begin, end] {
                ready.count_down();
                go.wait(false, std::memory_order_acquire);
                if (cancelled) return;
                std::uint64_t failures = 0;
                for (auto i = begin; i < end; ++i) {
                    stack.push(&nodes[i], failures);
                    popped[i] = stack.pop(failures);
                }
                stats[t].failures = failures;
                done.count_down();
            });
        }
    } catch (...) {
        cancelled = true;
        go.store(true, std::memory_order_release);
        go.notify_all();
        for (auto& worker : workers) worker.join();
        throw;
    }

    ready.wait();
    const double cpu_start = cpu_seconds();
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    go.notify_all();
    done.wait();
    const auto end = std::chrono::steady_clock::now();
    const double cpu_end = cpu_seconds();
    for (auto& worker : workers) worker.join();

    // Full per-node check, outside the timed region. A checksum alone can miss
    // duplicate/lost nodes. Every output must be one distinct allocated node.
    std::vector<bool> seen(pairs, false);
    for (Node* node : popped) {
        if (!node) throw std::runtime_error("unexpected empty pop");
        const auto id = node->id;
        if (id >= pairs || node != &nodes[id] || seen[id])
            throw std::runtime_error("invalid or duplicate node");
        seen[id] = true;
    }
    if (!stack.empty()) throw std::runtime_error("nonempty final stack");
    std::uint64_t failures = 0;
    for (const auto& stat : stats) failures += stat.failures;
    return {std::chrono::duration<double>(end - start).count(),
            cpu_end - cpu_start, failures};
}

std::size_t number(std::string_view value) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0)
        throw std::runtime_error("expected a positive integer");
    return result;
}

template<class Stack>
void check_lifo() {
    Stack stack;
    Node nodes[3];
    std::uint64_t failures = 0;
    if (stack.pop(failures) != nullptr) throw std::runtime_error("initial empty check failed");
    for (auto& node : nodes) stack.push(&node, failures);
    for (int i = 2; i >= 0; --i)
        if (stack.pop(failures) != &nodes[i]) throw std::runtime_error("LIFO check failed");
    if (!stack.empty() || stack.pop(failures)) throw std::runtime_error("final empty check failed");
}

#ifndef STACK_BENCHMARK_NO_MAIN
int main(int argc, char** argv) try {
    std::size_t pairs = 1'000'000;
    unsigned repeats = 7;
    unsigned seed = 20261017;
    std::vector<unsigned> counts{1, 2, 4, 8, 16, 32, 64, 128};
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--help") {
            std::cout << "Usage: benchmark [--pairs N] [--repeats N] [--seed N]"
                         " [--threads 1,2,4,8,16,32,64,128]\n";
            return 0;
        }
        if (++i >= argc) throw std::runtime_error("missing option value");
        const std::string_view value = argv[i];
        if (option == "--pairs") pairs = number(value);
        else if (option == "--repeats") {
            const auto n = number(value);
            if (n > 1000) throw std::runtime_error("repeats must be <= 1000");
            repeats = static_cast<unsigned>(n);
        } else if (option == "--seed") {
            const auto n = number(value);
            if (n > UINT32_MAX) throw std::runtime_error("seed must fit uint32");
            seed = static_cast<unsigned>(n);
        } else if (option == "--threads") {
            counts.clear();
            std::size_t pos = 0;
            do {
                const auto comma = value.find(',', pos);
                const auto n = number(value.substr(pos, comma - pos));
                if (n > 1024) throw std::runtime_error("threads must be <= 1024");
                counts.push_back(static_cast<unsigned>(n));
                if (comma == std::string_view::npos) break;
                pos = comma + 1;
            } while (true);
        } else throw std::runtime_error("unknown option: " + std::string(option));
    }
    if (pairs < *std::max_element(counts.begin(), counts.end()) || pairs > 100'000'000)
        throw std::runtime_error("pairs must be >= threads and <= 100000000");

    check_lifo<MutexStack>();
    check_lifo<TreiberStack>();
    std::cerr << "hardware_concurrency=" << std::thread::hardware_concurrency()
              << " pointer_atomic_lock_free=" << std::atomic<Node*>{}.is_lock_free()
              << " count_retries=" << STACK_COUNT_RETRIES
              << " seed=" << seed << "\n";
    std::cout << "round,order,threads,implementation,pairs,wall_seconds,cpu_seconds,"
                 "million_ops_per_second,cpu_per_wall,cas_failures,failures_per_operation\n";
    std::cout << std::fixed << std::setprecision(9);
    std::mt19937 random(seed);
    // Warm both implementations at every thread count before collecting data.
    for (const auto threads : counts) {
        run<MutexStack>(std::max<std::size_t>(10'000, threads), threads);
        run<TreiberStack>(std::max<std::size_t>(10'000, threads), threads);
    }
    // Shuffle thread counts each round and counterbalance implementation order.
    // Every configuration has the same total number of successful operations.
    for (unsigned round = 1; round <= repeats; ++round) {
        std::shuffle(counts.begin(), counts.end(), random);
        for (const unsigned threads : counts) {
            const bool cas_first = ((round + threads) % 2 == 0);
            for (unsigned order = 0; order < 2; ++order) {
                const bool cas = (order == 0) == cas_first;
                const auto result = cas ? run<TreiberStack>(pairs, threads)
                                        : run<MutexStack>(pairs, threads);
                const double operations = 2.0 * static_cast<double>(pairs);
                std::cout << round << ',' << order + 1 << ',' << threads << ','
                          << (cas ? "treiber" : "mutex") << ',' << pairs << ','
                          << result.wall_seconds << ',' << result.cpu_seconds << ','
                          << operations / result.wall_seconds / 1e6 << ','
                          << result.cpu_seconds / result.wall_seconds << ','
                          << result.failures << ',' << result.failures / operations << '\n';
                std::cout.flush();
            }
        }
    }
} catch (const std::exception& error) {
    std::cerr << "benchmark: " << error.what() << '\n';
    return 1;
}
#endif
