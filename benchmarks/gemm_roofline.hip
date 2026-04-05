// GEMM Roofline Benchmark — hipBLASLt across shapes × data types
//
// Sweeps square & LLM-inference shapes with FP16 / BF16 / FP32 / FP8
// to produce (Arithmetic Intensity, TFLOPS) data points for roofline charts.
//
// Build:  make                        (needs hipBLASLt)
// Run:    ./build/gemm_roofline                          # default shapes
//         ./build/gemm_roofline 4096,4096,4096           # custom shape
//         ./build/gemm_roofline 4096,4096,4096 1,4096,11008

#include "bench_common.h"
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#define BLASLT_CHECK(cmd) do { \
    hipblasStatus_t s = cmd; \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLASLt error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// =========================================================================
//  Shape definitions
// =========================================================================

struct GemmShape {
    int M, N, K;
    std::string tag;
};

static std::vector<GemmShape> default_shapes() {
    return {
        {128,   128,   128,   "sq-128"},
        {256,   256,   256,   "sq-256"},
        {512,   512,   512,   "sq-512"},
        {1024,  1024,  1024,  "sq-1k"},
        {2048,  2048,  2048,  "sq-2k"},
        {4096,  4096,  4096,  "sq-4k"},
        {8192,  8192,  8192,  "sq-8k"},
        {16384, 16384, 16384, "sq-16k"},
        {1,     4096,  4096,  "tok-1"},
        {32,    4096,  4096,  "tok-32"},
        {128,   4096,  4096,  "tok-128"},
        {512,   4096,  4096,  "pf-512"},
        {2048,  4096,  4096,  "pf-2k"},
        {4096,  11008, 4096,  "ffn-up"},
        {4096,  4096,  11008, "ffn-dn"},
    };
}

static bool parse_shape(const char* arg, GemmShape& out) {
    int m, n, k;
    if (sscanf(arg, "%d,%d,%d", &m, &n, &k) != 3) return false;
    if (m < 1 || n < 1 || k < 1) return false;
    char buf[64];
    snprintf(buf, sizeof(buf), "%dx%dx%d", m, n, k);
    out = {m, n, k, buf};
    return true;
}

static void usage(const char* prog) {
    printf("Usage: %s [M,N,K ...]\n\n", prog);
    printf("  No arguments  — run built-in shapes (square + LLM inference)\n");
    printf("  M,N,K          — benchmark one or more custom shapes\n\n");
    printf("Examples:\n");
    printf("  %s                                  # default 15 shapes\n", prog);
    printf("  %s 4096,4096,4096                   # single custom shape\n", prog);
    printf("  %s 4096,4096,4096 1,4096,11008      # multiple custom shapes\n", prog);
}

// =========================================================================
//  Data type definitions
// =========================================================================

struct DtypeDef {
    const char* name;
    hipDataType a_type, b_type, c_type;
    hipblasComputeType_t compute_type;
    hipDataType scale_type;
    int in_bytes;                   // bytes per A/B element
    int out_bytes;                  // bytes per C element
    double flops_per_cu_per_clock;  // MFMA throughput (CDNA3)
};

static DtypeDef dtypes[] = {
    {"FP16", HIP_R_16F,  HIP_R_16F,  HIP_R_16F,  HIPBLAS_COMPUTE_32F, HIP_R_32F, 2, 2, 2048},
    {"BF16", HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, HIP_R_32F, 2, 2, 2048},
    {"FP32", HIP_R_32F,  HIP_R_32F,  HIP_R_32F,  HIPBLAS_COMPUTE_32F, HIP_R_32F, 4, 4, 1024},
    {"FP8",  HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16F,
             HIPBLAS_COMPUTE_32F, HIP_R_32F, 1, 2, 4096},
};

static constexpr int NUM_DTYPES = sizeof(dtypes) / sizeof(dtypes[0]);

// =========================================================================
//  Main
// =========================================================================

int main(int argc, char** argv) {
    // Parse shapes from command line or use defaults
    std::vector<GemmShape> shapes;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        }
        GemmShape sh;
        if (!parse_shape(argv[i], sh)) {
            fprintf(stderr, "Bad shape '%s' — expected M,N,K (e.g. 4096,4096,4096)\n", argv[i]);
            return 1;
        }
        shapes.push_back(sh);
    }
    if (shapes.empty()) shapes = default_shapes();

    GpuInfo gpu = get_gpu_info();

    printf("=== GEMM Roofline Benchmark (hipBLASLt) ===\n");
    print_gpu_info(gpu);
    printf("Shapes:   %zu\n\n", shapes.size());

    // Pre-allocate device buffers based on largest shape
    size_t max_bytes = 0;
    for (auto& sh : shapes) {
        max_bytes = std::max(max_bytes, (size_t)sh.M * sh.K * 4);
        max_bytes = std::max(max_bytes, (size_t)sh.K * sh.N * 4);
        max_bytes = std::max(max_bytes, (size_t)sh.M * sh.N * 4);
    }
    const size_t MAX_BYTES = max_bytes;
    const size_t WS_BYTES  = 256ULL * 1024 * 1024;  // 256 MB workspace

    void *d_A, *d_B, *d_C, *d_ws;
    HIP_CHECK(hipMalloc(&d_A, MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_B, MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_C, MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_ws, WS_BYTES));
    HIP_CHECK(hipMemset(d_A, 0x3f, MAX_BYTES));
    HIP_CHECK(hipMemset(d_B, 0x3e, MAX_BYTES));
    HIP_CHECK(hipMemset(d_C, 0,    MAX_BYTES));

    hipblasLtHandle_t handle;
    BLASLT_CHECK(hipblasLtCreate(&handle));

    BenchEvents ev;
    hipStream_t stream = nullptr;
    float alpha_f = 1.0f, beta_f = 0.0f;

    for (int di = 0; di < NUM_DTYPES; di++) {
        auto& dt = dtypes[di];

        double peak_tflops = (double)gpu.prop.multiProcessorCount
                           * dt.flops_per_cu_per_clock
                           * gpu.prop.clockRate * 1e-9;

        printf("--- %s  (est. peak: %.1f TFLOPS) ---\n", dt.name, peak_tflops);
        printf("%-10s %6s %6s %6s  %8s %7s %6s %10s\n",
               "Tag", "M", "N", "K", "TFLOPS", "AI", "Eff%", "Time");
        printf("--------------------------------------------------------------\n");

        for (int si = 0; si < (int)shapes.size(); si++) {
            auto& sh = shapes[si];

            size_t a_bytes = (size_t)sh.M * sh.K * dt.in_bytes;
            size_t b_bytes = (size_t)sh.K * sh.N * dt.in_bytes;
            size_t c_bytes = (size_t)sh.M * sh.N * dt.out_bytes;
            if (a_bytes > MAX_BYTES || b_bytes > MAX_BYTES || c_bytes > MAX_BYTES) {
                printf("%-10s %6d %6d %6d  %8s\n", sh.tag.c_str(), sh.M, sh.N, sh.K, "OOM");
                continue;
            }

            // Create matmul descriptor
            hipblasLtMatmulDesc_t matmul;
            BLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul, dt.compute_type, dt.scale_type));
            hipblasOperation_t op_n = HIPBLAS_OP_N;
            BLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
            BLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

            // Create matrix layouts (column-major)
            hipblasLtMatrixLayout_t la, lb, lc;
            BLASLT_CHECK(hipblasLtMatrixLayoutCreate(&la, dt.a_type, sh.M, sh.K, sh.M));
            BLASLT_CHECK(hipblasLtMatrixLayoutCreate(&lb, dt.b_type, sh.K, sh.N, sh.K));
            BLASLT_CHECK(hipblasLtMatrixLayoutCreate(&lc, dt.c_type, sh.M, sh.N, sh.M));

            // Find best algorithm (try up to 5, quick-test each)
            hipblasLtMatmulPreference_t pref;
            BLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
            size_t ws = WS_BYTES;
            BLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

            const int MAX_ALGOS = 5;
            hipblasLtMatmulHeuristicResult_t hresults[MAX_ALGOS];
            int found = 0;
            hipblasStatus_t hstat = hipblasLtMatmulAlgoGetHeuristic(
                handle, matmul, la, lb, lc, lc, pref, MAX_ALGOS, hresults, &found);

            if (hstat != HIPBLAS_STATUS_SUCCESS || found == 0) {
                printf("%-10s %6d %6d %6d  %8s\n", sh.tag.c_str(), sh.M, sh.N, sh.K, "no algo");
                BLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
                BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(la));
                BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(lb));
                BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(lc));
                BLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
                continue;
            }

            double flops = 2.0 * sh.M * sh.N * sh.K;

            // Quick-test each algorithm (3 iters) to pick the fastest
            int best_ai = 0;
            double best_tf = 0;
            for (int ai = 0; ai < found; ai++) {
                hipblasLtMatmul(handle, matmul, &alpha_f, d_A, la, d_B, lb,
                                &beta_f, d_C, lc, d_C, lc,
                                &hresults[ai].algo, d_ws, WS_BYTES, stream);
                HIP_CHECK(hipDeviceSynchronize());

                double sum_tf = 0;
                for (int t = 0; t < 3; t++) {
                    HIP_CHECK(hipEventRecord(ev.start));
                    hipblasLtMatmul(handle, matmul, &alpha_f, d_A, la, d_B, lb,
                                    &beta_f, d_C, lc, d_C, lc,
                                    &hresults[ai].algo, d_ws, WS_BYTES, stream);
                    HIP_CHECK(hipEventRecord(ev.stop));
                    HIP_CHECK(hipEventSynchronize(ev.stop));
                    float ms;
                    HIP_CHECK(hipEventElapsedTime(&ms, ev.start, ev.stop));
                    sum_tf += flops / (ms * 1e9);
                }
                double avg = sum_tf / 3.0;
                if (avg > best_tf) { best_tf = avg; best_ai = ai; }
            }

            // Full benchmark with best algorithm
            const int WARMUP = 10, NITER = 50;
            for (int i = 0; i < WARMUP; i++)
                hipblasLtMatmul(handle, matmul, &alpha_f, d_A, la, d_B, lb,
                                &beta_f, d_C, lc, d_C, lc,
                                &hresults[best_ai].algo, d_ws, WS_BYTES, stream);
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<double> tv(NITER);
            for (int i = 0; i < NITER; i++) {
                HIP_CHECK(hipEventRecord(ev.start));
                hipblasLtMatmul(handle, matmul, &alpha_f, d_A, la, d_B, lb,
                                &beta_f, d_C, lc, d_C, lc,
                                &hresults[best_ai].algo, d_ws, WS_BYTES, stream);
                HIP_CHECK(hipEventRecord(ev.stop));
                HIP_CHECK(hipEventSynchronize(ev.stop));
                float ms;
                HIP_CHECK(hipEventElapsedTime(&ms, ev.start, ev.stop));
                tv[i] = flops / (ms * 1e9);   // TFLOPS
            }

            Stats s = compute_stats(tv);
            double total_data = (double)sh.M * sh.K * dt.in_bytes
                              + (double)sh.K * sh.N * dt.in_bytes
                              + (double)sh.M * sh.N * dt.out_bytes;
            double ai_val = flops / total_data;
            double eff    = s.avg / peak_tflops * 100.0;
            double avg_ms = flops / (s.avg * 1e9);

            printf("%-10s %6d %6d %6d  %8.1f %7.1f %5.1f%% %8.3fms\n",
                   sh.tag.c_str(), sh.M, sh.N, sh.K, s.avg, ai_val, eff, avg_ms);

            BLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
            BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(la));
            BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(lb));
            BLASLT_CHECK(hipblasLtMatrixLayoutDestroy(lc));
            BLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
        }

        printf("\n");
        if (di < NUM_DTYPES - 1) cooldown(3);
    }

    BLASLT_CHECK(hipblasLtDestroy(handle));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_ws));

    printf("Done.\n");
    return 0;
}
