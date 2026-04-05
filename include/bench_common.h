#pragma once

#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unistd.h>

// =========================================================================
//  Error checking
// =========================================================================

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d: %s\n", \
                e, __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// =========================================================================
//  GPU information
// =========================================================================

struct GpuInfo {
    hipDeviceProp_t prop;
    size_t l2_reported;
    size_t l2_estimated;
};

inline GpuInfo get_gpu_info(int device = 0) {
    GpuInfo info;
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipGetDeviceProperties(&info.prop, device));
    info.l2_reported  = info.prop.l2CacheSize;
    info.l2_estimated = info.l2_reported * info.prop.multiProcessorCount / 10;
    return info;
}

inline void print_gpu_info(const GpuInfo& info) {
    printf("GPU:      %s\n", info.prop.name);
    printf("CUs:      %d\n", info.prop.multiProcessorCount);
    printf("Clock:    %d MHz\n", info.prop.clockRate / 1000);
    printf("HBM:      %zu MB\n", info.prop.totalGlobalMem >> 20);
    printf("L2 Cache: %zu KB (reported), ~%zu MB (estimated total)\n",
           info.l2_reported >> 10, info.l2_estimated >> 20);
}

// =========================================================================
//  Array size helpers
// =========================================================================

// Returns N (number of floats) that is guaranteed to stress HBM, not L2.
// alignment_elems: total elements divisible by this (for kernel vectorization).
inline unsigned int auto_array_size(const GpuInfo& info,
                                    unsigned int alignment_elems = 32768) {
    size_t min_bytes = std::max(info.l2_estimated * 4,
                                (size_t)(2ULL * 1024 * 1024 * 1024));
    unsigned int N = (unsigned int)(min_bytes / sizeof(float));
    N = (N / alignment_elems) * alignment_elems;
    return N;
}

// =========================================================================
//  Statistics
// =========================================================================

struct Stats {
    double avg, mx, mn, median, p5, p95;
};

inline Stats compute_stats(std::vector<double>& data) {
    Stats s{};
    if (data.empty()) return s;
    std::sort(data.begin(), data.end());
    double sum = 0;
    for (auto v : data) sum += v;
    s.avg    = sum / data.size();
    s.mn     = data.front();
    s.mx     = data.back();
    s.median = data[data.size() / 2];
    s.p5     = data[(size_t)(data.size() * 0.05)];
    s.p95    = data[(size_t)(data.size() * 0.95)];
    return s;
}

// =========================================================================
//  Benchmark helpers
// =========================================================================

struct BenchEvents {
    hipEvent_t start, stop;

    BenchEvents() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchEvents() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    BenchEvents(const BenchEvents&) = delete;
    BenchEvents& operator=(const BenchEvents&) = delete;
};

// Quick average — returns mean bandwidth (GB/s).
template<typename F>
double bench_avg(BenchEvents& ev, F launch_fn, size_t data_bytes,
                 int niter, int warmup = 5) {
    for (int i = 0; i < warmup; i++) launch_fn();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> bws(niter);
    for (int i = 0; i < niter; i++) {
        HIP_CHECK(hipEventRecord(ev.start));
        launch_fn();
        HIP_CHECK(hipEventRecord(ev.stop));
        HIP_CHECK(hipEventSynchronize(ev.stop));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, ev.start, ev.stop));
        bws[i] = data_bytes / (ms * 1e6);
    }
    auto s = compute_stats(bws);
    return s.avg;
}

// Full stats — returns detailed statistics.
template<typename F>
Stats bench_full(BenchEvents& ev, F launch_fn, size_t data_bytes,
                 int niter, int warmup = 5) {
    for (int i = 0; i < warmup; i++) launch_fn();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> bws(niter);
    for (int i = 0; i < niter; i++) {
        HIP_CHECK(hipEventRecord(ev.start));
        launch_fn();
        HIP_CHECK(hipEventRecord(ev.stop));
        HIP_CHECK(hipEventSynchronize(ev.stop));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, ev.start, ev.stop));
        bws[i] = data_bytes / (ms * 1e6);
    }
    return compute_stats(bws);
}

// Measure latency instead of bandwidth — returns Stats in microseconds.
template<typename F>
Stats bench_latency_us(BenchEvents& ev, F launch_fn,
                       int niter, int warmup = 5) {
    for (int i = 0; i < warmup; i++) launch_fn();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> lats(niter);
    for (int i = 0; i < niter; i++) {
        HIP_CHECK(hipEventRecord(ev.start));
        launch_fn();
        HIP_CHECK(hipEventRecord(ev.stop));
        HIP_CHECK(hipEventSynchronize(ev.stop));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, ev.start, ev.stop));
        lats[i] = ms * 1000.0;
    }
    return compute_stats(lats);
}

// =========================================================================
//  Output formatting
// =========================================================================

inline void print_bw_result(const char* label, Stats& burst, Stats& sust) {
    printf("  %-12s burst=%7.1f  sustained=%7.1f  "
           "(p95=%7.1f  max=%7.1f  min=%7.1f)\n",
           label, burst.avg, sust.avg, sust.p95, sust.mx, sust.mn);
}

inline void print_latency_result(const char* label, Stats& s) {
    printf("  %-12s avg=%7.2f  p5=%7.2f  p95=%7.2f  "
           "min=%7.2f  max=%7.2f  (us)\n",
           label, s.avg, s.p5, s.p95, s.mn, s.mx);
}

// Cooldown helper — synchronize and sleep.
inline void cooldown(int seconds = 3) {
    HIP_CHECK(hipDeviceSynchronize());
    sleep(seconds);
}
