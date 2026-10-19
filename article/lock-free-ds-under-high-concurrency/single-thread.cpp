// macOS-only follow-up. Reuse the exact stack implementations and fresh-thread
// harness; suppress only the original command-line entry point.
#define STACK_BENCHMARK_NO_MAIN
#include "benchmark.cpp"
#include <array>
#include <pthread/qos.h>
#include <time.h>

#ifndef STACK_COLLECT_RUSAGE
#define STACK_COLLECT_RUSAGE 0
#endif
#if STACK_COLLECT_RUSAGE
#include <libproc.h>
#include <unistd.h>
struct Counters {
    std::uint64_t instructions, cycles, p_instructions, p_cycles;
};
thread_local Counters last_counts{};

rusage_info_v6 read_counts() {
    rusage_info_v6 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V6,
                       reinterpret_cast<rusage_info_t*>(&info)) != 0)
        throw std::runtime_error("reading process counters failed");
    return info;
}
#endif

double thread_cpu_seconds() {
    timespec time{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) != 0)
        throw std::runtime_error("thread CPU clock failed");
    return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9;
}

// Unlike the concurrent benchmark, this function is called by ONE worker only.
// Reusing nodes is safe here: no other reader exists between push/pop pairs.
// Both implementations use the same arrays, and every timed loop starts empty.
template<class Stack>
__attribute__((noinline)) Result serial_measure(Stack& stack, Node* nodes,
                                              Node** popped, std::size_t pairs,
                                              unsigned passes) {
    std::uint64_t failures = 0;
#if STACK_COLLECT_RUSAGE
    const auto before = read_counts();
#endif
    const double cpu_start = thread_cpu_seconds();
    const auto start = std::chrono::steady_clock::now();
    for (unsigned pass = 0; pass < passes; ++pass) {
        for (std::size_t i = 0; i < pairs; ++i) {
            stack.push(&nodes[i], failures);
            popped[i] = stack.pop(failures);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double cpu_end = thread_cpu_seconds();
#if STACK_COLLECT_RUSAGE
    const auto after = read_counts();
    // Process-wide deltas; the main thread is joining this worker and does not
    // actively drive the test. Clock/API overhead is included in these deltas.
    last_counts = {after.ri_instructions - before.ri_instructions,
                   after.ri_cycles - before.ri_cycles,
                   after.ri_pinstructions - before.ri_pinstructions,
                   after.ri_pcycles - before.ri_pcycles};
#endif
    if (!stack.empty()) throw std::runtime_error("serial stack is not empty");
    for (std::size_t i = 0; i < pairs; ++i)
        if (popped[i] != &nodes[i]) throw std::runtime_error("serial node check failed");
    return {std::chrono::duration<double>(end - start).count(),
            cpu_end - cpu_start, failures};
}

void print_result(std::string_view experiment, std::string_view qos,
                  std::size_t pairs, unsigned passes, unsigned block,
                  unsigned position, bool cas, int actual_qos, const Result& r,
                  std::string_view implementation = {}) {
    const double operations = 2.0 * static_cast<double>(pairs) * passes;
    std::cout << experiment << ',' << qos << ',' << pairs << ',' << passes << ','
              << block << ',' << position << ','
              << (implementation.empty() ? (cas ? "treiber" : "mutex") : implementation) << ','
              << r.wall_seconds << ',' << r.cpu_seconds << ','
              << operations / r.wall_seconds / 1e6 << ',' << r.failures << ','
              << actual_qos;
#if STACK_COLLECT_RUSAGE
    const bool collected = experiment == "persistent" || experiment == "atomic_variants";
    std::cout << ',' << collected << ',' << (collected ? last_counts.instructions : 0)
              << ',' << (collected ? last_counts.cycles : 0)
              << ',' << (collected ? last_counts.p_instructions : 0)
              << ',' << (collected ? last_counts.p_cycles : 0);
#endif
    std::cout << '\n';
    std::cout.flush();
}

void persistent(std::string_view label, qos_class_t requested) {
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            if (requested != QOS_CLASS_UNSPECIFIED &&
                pthread_set_qos_class_self_np(requested, 0) != 0)
                throw std::runtime_error("setting QoS failed");
            qos_class_t actual{};
            int priority = 0;
            if (pthread_get_qos_class_np(pthread_self(), &actual, &priority) != 0)
                throw std::runtime_error("reading QoS failed");
            constexpr std::size_t pairs = 1'000'000;
            auto nodes = std::make_unique<Node[]>(pairs);
            std::vector<Node*> popped(pairs);
            for (std::size_t i = 0; i < pairs; ++i) nodes[i].id = i;
            MutexStack mutex;
            TreiberStack treiber;
            // Warm the SAME worker and buffers that will collect the samples.
            serial_measure(mutex, nodes.get(), popped.data(), pairs, 2);
            serial_measure(treiber, nodes.get(), popped.data(), pairs, 2);
            struct Job { unsigned passes, block; };
            std::vector<Job> jobs;
            for (unsigned block = 1; block <= 6; ++block)
                for (unsigned passes : {1u, 8u}) jobs.push_back({passes, block});
            std::mt19937 random(20261018);
            std::shuffle(jobs.begin(), jobs.end(), random);
            for (const auto [passes, block] : jobs) {
                // ABBA / BAAB counterbalance the position of each implementation.
                const std::array<bool, 4> order = block % 2 == 1
                    ? std::array<bool, 4>{false, true, true, false}
                    : std::array<bool, 4>{true, false, false, true};
                for (unsigned position = 0; position < order.size(); ++position) {
                    const bool cas = order[position];
                    const auto result = cas
                        ? serial_measure(treiber, nodes.get(), popped.data(), pairs, passes)
                        : serial_measure(mutex, nodes.get(), popped.data(), pairs, passes);
                    print_result("persistent", label, pairs, passes, block,
                                 position + 1, cas, static_cast<int>(actual), result);
                }
            }
        } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    if (error) std::rethrow_exception(error);
}

// Diagnostic controls ONLY. These are deliberately NOT concurrent stacks:
// relaxed publication is insufficient for general multi-threaded use, and the
// load/store variant cannot make concurrent head updates indivisible.
template<bool ReleasePush, bool AcquirePop>
class SingleThreadDiagnosticCAS {
    std::atomic<Node*> head_{nullptr};
public:
    void push(Node* node, std::uint64_t& failures) {
        Node* expected = head_.load(std::memory_order_relaxed);
        do {
            node->next = expected;
            if (head_.compare_exchange_weak(expected, node,
                    ReleasePush ? std::memory_order_release : std::memory_order_relaxed,
                    std::memory_order_relaxed)) return;
            ++failures;
        } while (true);
    }
    Node* pop(std::uint64_t& failures) {
        constexpr auto order = AcquirePop ? std::memory_order_acquire : std::memory_order_relaxed;
        Node* expected = head_.load(order);
        while (expected) {
            Node* next = expected->next;
            if (head_.compare_exchange_weak(expected, next, order, order)) return expected;
            ++failures;
        }
        return nullptr;
    }
    bool empty() const { return head_.load(std::memory_order_relaxed) == nullptr; }
};

class SingleThreadAtomicLoadStore {
    std::atomic<Node*> head_{nullptr};
public:
    void push(Node* node, std::uint64_t&) {
        node->next = head_.load(std::memory_order_relaxed);
        head_.store(node, std::memory_order_relaxed);
    }
    Node* pop(std::uint64_t&) {
        Node* node = head_.load(std::memory_order_relaxed);
        if (node) head_.store(node->next, std::memory_order_relaxed);
        return node;
    }
    bool empty() const { return head_.load(std::memory_order_relaxed) == nullptr; }
};

void atomic_variants() {
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) != 0)
                throw std::runtime_error("setting QoS failed");
            qos_class_t actual{};
            int priority = 0;
            if (pthread_get_qos_class_np(pthread_self(), &actual, &priority) != 0)
                throw std::runtime_error("reading QoS failed");
            constexpr std::size_t pairs = 1'000'000;
            auto nodes = std::make_unique<Node[]>(pairs);
            std::vector<Node*> popped(pairs);
            MutexStack mutex;
            TreiberStack treiber;
            SingleThreadDiagnosticCAS<false, false> relaxed;
            SingleThreadDiagnosticCAS<true, false> release_push;
            SingleThreadDiagnosticCAS<false, true> acquire_pop;
            SingleThreadAtomicLoadStore load_store;
            const auto measure = [&](unsigned kind, unsigned passes) {
                switch (kind) {
                case 0: return serial_measure(mutex, nodes.get(), popped.data(), pairs, passes);
                case 1: return serial_measure(treiber, nodes.get(), popped.data(), pairs, passes);
                case 2: return serial_measure(relaxed, nodes.get(), popped.data(), pairs, passes);
                case 3: return serial_measure(release_push, nodes.get(), popped.data(), pairs, passes);
                case 4: return serial_measure(acquire_pop, nodes.get(), popped.data(), pairs, passes);
                default: return serial_measure(load_store, nodes.get(), popped.data(), pairs, passes);
                }
            };
            const std::array<std::string_view, 6> labels{
                "mutex", "treiber", "relaxed_cas_SINGLE_THREAD_ONLY",
                "release_push_ONLY_SINGLE_THREAD_ONLY", "acquire_pop_ONLY_SINGLE_THREAD_ONLY",
                "relaxed_load_store_SINGLE_THREAD_ONLY"};
            for (unsigned kind = 0; kind < labels.size(); ++kind) measure(kind, 2);
            std::mt19937 random(20261018);
            // Cyclic permutations put each implementation in every position;
            // shuffle the blocks to avoid a fixed progression through positions.
            std::vector<unsigned> blocks(12);
            std::iota(blocks.begin(), blocks.end(), 0);
            std::shuffle(blocks.begin(), blocks.end(), random);
            for (const auto block : blocks) {
                for (unsigned position = 0; position < labels.size(); ++position) {
                    const auto kind = (block + position) % labels.size();
                    const auto result = measure(kind, 8);
                    print_result("atomic_variants", "user_initiated", pairs, 8,
                                 block + 1, position + 1, kind != 0,
                                 static_cast<int>(actual), result, labels[kind]);
                }
            }
        } catch (...) { error = std::current_exception(); }
    });
    worker.join();
    if (error) std::rethrow_exception(error);
}

int main(int argc, char** argv) try {
    std::cout << "experiment,qos,pairs,passes,block,position,implementation,"
                 "wall_seconds,cpu_seconds,million_ops_per_second,cas_failures,actual_qos";
#if STACK_COLLECT_RUSAGE
    std::cout << ",counters_collected,instructions,cycles,p_instructions,p_cycles";
#endif
    std::cout << '\n';
    std::cout << std::fixed << std::setprecision(9);
    check_lifo<MutexStack>();
    check_lifo<TreiberStack>();
    if (argc == 2 && std::string_view(argv[1]) == "--atomic-variants") {
        atomic_variants();
        return 0;
    }
    if (argc != 1) throw std::runtime_error("usage: single-thread [--atomic-variants]");

    // Empty-loop timing estimates start/finish synchronization overhead only.
    run<MutexStack>(0, 1);
    for (unsigned block = 1; block <= 20; ++block)
        print_result("wake_only", "inherited", 0, 1, block, 1, false, -1,
                     run<MutexStack>(0, 1));

    run<MutexStack>(10'000, 1);
    run<TreiberStack>(10'000, 1);
    for (const std::size_t pairs : {1'000'000u, 8'000'000u}) {
        for (unsigned block = 1; block <= 12; ++block) {
            for (unsigned position = 0; position < 2; ++position) {
                const bool cas = (position == 0) == (block % 2 == 1);
                const auto result = cas ? run<TreiberStack>(pairs, 1)
                                        : run<MutexStack>(pairs, 1);
                print_result("fresh", "inherited", pairs, 1, block,
                             position + 1, cas, -1, result);
            }
        }
    }
    persistent("inherited", QOS_CLASS_UNSPECIFIED);
    persistent("user_initiated", QOS_CLASS_USER_INITIATED);
    persistent("background", QOS_CLASS_BACKGROUND);
} catch (const std::exception& error) {
    std::cerr << "single-thread: " << error.what() << '\n';
    return 1;
}
