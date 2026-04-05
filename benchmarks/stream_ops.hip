// Full STREAM + HBM Benchmark for AMD Instinct GPUs
//
// ISA-optimised: v4f (128-bit) non-temporal loads/stores, auto-tuning
// TB (thread block size) x MLP (memory-level parallelism) sweep.
//
// Operations & data movement:
//   Read:  sink  += a[i]                — 1R      = 1x array
//   Write: c[i]  = 0                    — 1W      = 1x array
//   Copy:  c[i]  = a[i]                 — 1R + 1W = 2x array
//   Mul:   c[i]  = scalar * a[i]        — 1R + 1W = 2x array
//   Add:   c[i]  = a[i] + b[i]          — 2R + 1W = 3x array
//   Triad: c[i]  = a[i] + scalar*b[i]   — 2R + 1W = 3x array (FMA)
//   Dot:   sum  += a[i] * b[i]          — 2R      = 2x array
//
// Build:  make
// Run:    HIP_FORCE_DEV_KERNARG=1 ./build/stream_ops
//         HIP_FORCE_DEV_KERNARG=1 ./build/stream_ops 671088640

#include "bench_common.h"
#include "types.h"
#include <cfloat>

// =========================================================================
//  Read kernels: sink += a[i]
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void read_kern(const float* __restrict a, float* __restrict c, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;
    v4f acc = {0,0,0,0};

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride)
            acc += __builtin_nontemporal_load(a4 + i);
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            acc += v0 + v1;
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            acc += v0 + v1 + v2 + v3;
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f v4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f v5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f v6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f v7 = __builtin_nontemporal_load(a4 + i + stride*7);
            acc += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
        }
    }
    float t = acc[0] + acc[1] + acc[2] + acc[3];
    if (t == FLT_MIN) c[tid] = t;
}

// =========================================================================
//  Write kernels: c[i] = 0
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void write_kern(float* __restrict c, unsigned int N) {
    v4f* c4 = reinterpret_cast<v4f*>(c);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;
    v4f val = {0.f, 0.f, 0.f, 0.f};

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride)
            __builtin_nontemporal_store(val, c4 + i);
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            __builtin_nontemporal_store(val, c4 + i);
            __builtin_nontemporal_store(val, c4 + i + stride);
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            __builtin_nontemporal_store(val, c4 + i);
            __builtin_nontemporal_store(val, c4 + i + stride);
            __builtin_nontemporal_store(val, c4 + i + stride*2);
            __builtin_nontemporal_store(val, c4 + i + stride*3);
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            __builtin_nontemporal_store(val, c4 + i);
            __builtin_nontemporal_store(val, c4 + i + stride);
            __builtin_nontemporal_store(val, c4 + i + stride*2);
            __builtin_nontemporal_store(val, c4 + i + stride*3);
            __builtin_nontemporal_store(val, c4 + i + stride*4);
            __builtin_nontemporal_store(val, c4 + i + stride*5);
            __builtin_nontemporal_store(val, c4 + i + stride*6);
            __builtin_nontemporal_store(val, c4 + i + stride*7);
        }
    }
}

// =========================================================================
//  Copy kernels: c[i] = a[i]
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void copy_kern(const float* __restrict a, float* __restrict c, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    v4f* c4 = reinterpret_cast<v4f*>(c);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride) {
            v4f v = __builtin_nontemporal_load(a4 + i);
            __builtin_nontemporal_store(v, c4 + i);
        }
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            __builtin_nontemporal_store(v0, c4 + i);
            __builtin_nontemporal_store(v1, c4 + i + stride);
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            __builtin_nontemporal_store(v0, c4 + i);
            __builtin_nontemporal_store(v1, c4 + i + stride);
            __builtin_nontemporal_store(v2, c4 + i + stride*2);
            __builtin_nontemporal_store(v3, c4 + i + stride*3);
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f v4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f v5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f v6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f v7 = __builtin_nontemporal_load(a4 + i + stride*7);
            __builtin_nontemporal_store(v0, c4 + i);
            __builtin_nontemporal_store(v1, c4 + i + stride);
            __builtin_nontemporal_store(v2, c4 + i + stride*2);
            __builtin_nontemporal_store(v3, c4 + i + stride*3);
            __builtin_nontemporal_store(v4, c4 + i + stride*4);
            __builtin_nontemporal_store(v5, c4 + i + stride*5);
            __builtin_nontemporal_store(v6, c4 + i + stride*6);
            __builtin_nontemporal_store(v7, c4 + i + stride*7);
        }
    }
}

// =========================================================================
//  Mul kernels: c[i] = scalar * a[i]
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void mul_kern(const float* __restrict a, float* __restrict c,
                         float scalar, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    v4f* c4 = reinterpret_cast<v4f*>(c);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;
    v4f s = {scalar, scalar, scalar, scalar};

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride) {
            v4f v = __builtin_nontemporal_load(a4 + i);
            __builtin_nontemporal_store(v * s, c4 + i);
        }
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            __builtin_nontemporal_store(v0 * s, c4 + i);
            __builtin_nontemporal_store(v1 * s, c4 + i + stride);
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            __builtin_nontemporal_store(v0 * s, c4 + i);
            __builtin_nontemporal_store(v1 * s, c4 + i + stride);
            __builtin_nontemporal_store(v2 * s, c4 + i + stride*2);
            __builtin_nontemporal_store(v3 * s, c4 + i + stride*3);
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f v0 = __builtin_nontemporal_load(a4 + i);
            v4f v1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f v2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f v3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f v4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f v5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f v6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f v7 = __builtin_nontemporal_load(a4 + i + stride*7);
            __builtin_nontemporal_store(v0 * s, c4 + i);
            __builtin_nontemporal_store(v1 * s, c4 + i + stride);
            __builtin_nontemporal_store(v2 * s, c4 + i + stride*2);
            __builtin_nontemporal_store(v3 * s, c4 + i + stride*3);
            __builtin_nontemporal_store(v4 * s, c4 + i + stride*4);
            __builtin_nontemporal_store(v5 * s, c4 + i + stride*5);
            __builtin_nontemporal_store(v6 * s, c4 + i + stride*6);
            __builtin_nontemporal_store(v7 * s, c4 + i + stride*7);
        }
    }
}

// =========================================================================
//  Add kernels: c[i] = a[i] + b[i]
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void add_kern(const float* __restrict a, const float* __restrict b,
                         float* __restrict c, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    const v4f* b4 = reinterpret_cast<const v4f*>(b);
    v4f* c4 = reinterpret_cast<v4f*>(c);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride) {
            v4f va = __builtin_nontemporal_load(a4 + i);
            v4f vb = __builtin_nontemporal_load(b4 + i);
            __builtin_nontemporal_store(va + vb, c4 + i);
        }
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            __builtin_nontemporal_store(va0 + vb0, c4 + i);
            __builtin_nontemporal_store(va1 + vb1, c4 + i + stride);
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            __builtin_nontemporal_store(va0 + vb0, c4 + i);
            __builtin_nontemporal_store(va1 + vb1, c4 + i + stride);
            __builtin_nontemporal_store(va2 + vb2, c4 + i + stride*2);
            __builtin_nontemporal_store(va3 + vb3, c4 + i + stride*3);
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f va4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f va5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f va6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f va7 = __builtin_nontemporal_load(a4 + i + stride*7);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            v4f vb4 = __builtin_nontemporal_load(b4 + i + stride*4);
            v4f vb5 = __builtin_nontemporal_load(b4 + i + stride*5);
            v4f vb6 = __builtin_nontemporal_load(b4 + i + stride*6);
            v4f vb7 = __builtin_nontemporal_load(b4 + i + stride*7);
            __builtin_nontemporal_store(va0 + vb0, c4 + i);
            __builtin_nontemporal_store(va1 + vb1, c4 + i + stride);
            __builtin_nontemporal_store(va2 + vb2, c4 + i + stride*2);
            __builtin_nontemporal_store(va3 + vb3, c4 + i + stride*3);
            __builtin_nontemporal_store(va4 + vb4, c4 + i + stride*4);
            __builtin_nontemporal_store(va5 + vb5, c4 + i + stride*5);
            __builtin_nontemporal_store(va6 + vb6, c4 + i + stride*6);
            __builtin_nontemporal_store(va7 + vb7, c4 + i + stride*7);
        }
    }
}

// =========================================================================
//  Triad kernels: c[i] = a[i] + scalar * b[i]   (FMA)
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void triad_kern(const float* __restrict a, const float* __restrict b,
                           float* __restrict c, float scalar, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    const v4f* b4 = reinterpret_cast<const v4f*>(b);
    v4f* c4 = reinterpret_cast<v4f*>(c);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;
    v4f s = {scalar, scalar, scalar, scalar};

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride) {
            v4f va = __builtin_nontemporal_load(a4 + i);
            v4f vb = __builtin_nontemporal_load(b4 + i);
            __builtin_nontemporal_store(va + s * vb, c4 + i);
        }
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            __builtin_nontemporal_store(va0 + s * vb0, c4 + i);
            __builtin_nontemporal_store(va1 + s * vb1, c4 + i + stride);
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            __builtin_nontemporal_store(va0 + s * vb0, c4 + i);
            __builtin_nontemporal_store(va1 + s * vb1, c4 + i + stride);
            __builtin_nontemporal_store(va2 + s * vb2, c4 + i + stride*2);
            __builtin_nontemporal_store(va3 + s * vb3, c4 + i + stride*3);
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f va4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f va5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f va6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f va7 = __builtin_nontemporal_load(a4 + i + stride*7);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            v4f vb4 = __builtin_nontemporal_load(b4 + i + stride*4);
            v4f vb5 = __builtin_nontemporal_load(b4 + i + stride*5);
            v4f vb6 = __builtin_nontemporal_load(b4 + i + stride*6);
            v4f vb7 = __builtin_nontemporal_load(b4 + i + stride*7);
            __builtin_nontemporal_store(va0 + s * vb0, c4 + i);
            __builtin_nontemporal_store(va1 + s * vb1, c4 + i + stride);
            __builtin_nontemporal_store(va2 + s * vb2, c4 + i + stride*2);
            __builtin_nontemporal_store(va3 + s * vb3, c4 + i + stride*3);
            __builtin_nontemporal_store(va4 + s * vb4, c4 + i + stride*4);
            __builtin_nontemporal_store(va5 + s * vb5, c4 + i + stride*5);
            __builtin_nontemporal_store(va6 + s * vb6, c4 + i + stride*6);
            __builtin_nontemporal_store(va7 + s * vb7, c4 + i + stride*7);
        }
    }
}

// =========================================================================
//  Dot kernels: sum += a[i] * b[i]   (pure 2-array read)
// =========================================================================

template<int TB, int MLP>
__launch_bounds__(TB)
__global__ void dot_kern(const float* __restrict a, const float* __restrict b,
                         float* __restrict scratch, unsigned int N) {
    const v4f* a4 = reinterpret_cast<const v4f*>(a);
    const v4f* b4 = reinterpret_cast<const v4f*>(b);
    unsigned int tid = threadIdx.x + blockIdx.x * TB;
    unsigned int stride = gridDim.x * TB;
    unsigned int n4 = N >> 2;
    v4f acc = {0, 0, 0, 0};

    if constexpr (MLP == 1) {
        for (unsigned int i = tid; i < n4; i += stride) {
            v4f va = __builtin_nontemporal_load(a4 + i);
            v4f vb = __builtin_nontemporal_load(b4 + i);
            acc += va * vb;
        }
    } else if constexpr (MLP == 2) {
        for (unsigned int i = tid; i + stride < n4; i += stride * 2) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            acc += va0 * vb0 + va1 * vb1;
        }
    } else if constexpr (MLP == 4) {
        for (unsigned int i = tid; i + stride*3 < n4; i += stride * 4) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            acc += va0 * vb0 + va1 * vb1 + va2 * vb2 + va3 * vb3;
        }
    } else if constexpr (MLP == 8) {
        for (unsigned int i = tid; i + stride*7 < n4; i += stride * 8) {
            v4f va0 = __builtin_nontemporal_load(a4 + i);
            v4f va1 = __builtin_nontemporal_load(a4 + i + stride);
            v4f va2 = __builtin_nontemporal_load(a4 + i + stride*2);
            v4f va3 = __builtin_nontemporal_load(a4 + i + stride*3);
            v4f va4 = __builtin_nontemporal_load(a4 + i + stride*4);
            v4f va5 = __builtin_nontemporal_load(a4 + i + stride*5);
            v4f va6 = __builtin_nontemporal_load(a4 + i + stride*6);
            v4f va7 = __builtin_nontemporal_load(a4 + i + stride*7);
            v4f vb0 = __builtin_nontemporal_load(b4 + i);
            v4f vb1 = __builtin_nontemporal_load(b4 + i + stride);
            v4f vb2 = __builtin_nontemporal_load(b4 + i + stride*2);
            v4f vb3 = __builtin_nontemporal_load(b4 + i + stride*3);
            v4f vb4 = __builtin_nontemporal_load(b4 + i + stride*4);
            v4f vb5 = __builtin_nontemporal_load(b4 + i + stride*5);
            v4f vb6 = __builtin_nontemporal_load(b4 + i + stride*6);
            v4f vb7 = __builtin_nontemporal_load(b4 + i + stride*7);
            acc += va0*vb0 + va1*vb1 + va2*vb2 + va3*vb3
                 + va4*vb4 + va5*vb5 + va6*vb6 + va7*vb7;
        }
    }

    float t = acc[0] + acc[1] + acc[2] + acc[3];
    if (t == FLT_MIN) scratch[tid] = t;
}

// =========================================================================
//  Kernel dispatch table
// =========================================================================

typedef void (*read_fn_t)(const float*, float*, unsigned int);
typedef void (*write_fn_t)(float*, unsigned int);
typedef void (*copy_fn_t)(const float*, float*, unsigned int);
typedef void (*mul_fn_t)(const float*, float*, float, unsigned int);
typedef void (*add_fn_t)(const float*, const float*, float*, unsigned int);
typedef void (*triad_fn_t)(const float*, const float*, float*, float, unsigned int);
typedef void (*dot_fn_t)(const float*, const float*, float*, unsigned int);

struct Config {
    int tb, mlp;
    read_fn_t  read;
    write_fn_t write;
    copy_fn_t  copy;
    mul_fn_t   mul;
    add_fn_t   add;
    triad_fn_t triad;
    dot_fn_t   dot;
};

#define REG(TB,MLP) { TB, MLP, \
    read_kern<TB,MLP>, write_kern<TB,MLP>, copy_kern<TB,MLP>, \
    mul_kern<TB,MLP>, add_kern<TB,MLP>, triad_kern<TB,MLP>, dot_kern<TB,MLP> }

static Config configs[] = {
    REG(64,1),  REG(64,2),  REG(64,4),  REG(64,8),
    REG(128,1), REG(128,2), REG(128,4), REG(128,8),
    REG(256,1), REG(256,2), REG(256,4), REG(256,8),
    REG(512,1), REG(512,2), REG(512,4), REG(512,8),
    REG(1024,1),REG(1024,2),REG(1024,4),REG(1024,8),
};

static constexpr int NCONFIGS = sizeof(configs) / sizeof(configs[0]);

// =========================================================================
//  Operation launch helpers — uniform signature for generic sweep/validate
// =========================================================================

typedef void (*op_launch_t)(Config&, int, float*, float*, float*, float, unsigned int);

static void do_read (Config& c,int g,float* a,float*  ,float* cc,float  ,unsigned int N){ c.read <<<dim3(g),dim3(c.tb)>>>(a,cc,N); }
static void do_write(Config& c,int g,float*  ,float*  ,float* cc,float  ,unsigned int N){ c.write<<<dim3(g),dim3(c.tb)>>>(cc,N); }
static void do_copy (Config& c,int g,float* a,float*  ,float* cc,float  ,unsigned int N){ c.copy <<<dim3(g),dim3(c.tb)>>>(a,cc,N); }
static void do_mul  (Config& c,int g,float* a,float*  ,float* cc,float s,unsigned int N){ c.mul  <<<dim3(g),dim3(c.tb)>>>(a,cc,s,N); }
static void do_add  (Config& c,int g,float* a,float* b,float* cc,float  ,unsigned int N){ c.add  <<<dim3(g),dim3(c.tb)>>>(a,b,cc,N); }
static void do_triad(Config& c,int g,float* a,float* b,float* cc,float s,unsigned int N){ c.triad<<<dim3(g),dim3(c.tb)>>>(a,b,cc,s,N); }
static void do_dot  (Config& c,int g,float* a,float* b,float* cc,float  ,unsigned int N){ c.dot  <<<dim3(g),dim3(c.tb)>>>(a,b,cc,N); }

struct OpDef {
    const char* name;
    const char* desc;
    size_t      data_bytes;
    op_launch_t launch;
};

// =========================================================================
//  Main
// =========================================================================

int main(int argc, char** argv) {
    GpuInfo gpu = get_gpu_info();

    unsigned int N = auto_array_size(gpu);
    if (argc > 1) {
        N = (unsigned int)atol(argv[1]);
        N = (N / 32768) * 32768;
    }

    size_t bytes = (size_t)N * sizeof(float);
    const float scalar = 0.4f;

    OpDef ops[] = {
        { "Read",  "1R",    bytes,     do_read  },
        { "Write", "1W",    bytes,     do_write },
        { "Copy",  "1R+1W", 2*bytes,   do_copy  },
        { "Mul",   "1R+1W", 2*bytes,   do_mul   },
        { "Add",   "2R+1W", 3*bytes,   do_add   },
        { "Triad", "2R+1W", 3*bytes,   do_triad },
        { "Dot",   "2R",    2*bytes,   do_dot   },
    };
    constexpr int NUM_OPS = 7;

    printf("=== GPU Memory Bandwidth Benchmark ===\n");
    print_gpu_info(gpu);
    printf("Array:    %u floats = %.2f GB  (x3 arrays = %.2f GB)\n",
           N, bytes / 1e9, bytes * 3 / 1e9);
    printf("Sweep:    %d configs (TB x MLP)\n", NCONFIGS);
    printf("Method:   each operation swept independently, 3s GPU cooldown between\n");
    printf("\nOperations & data moved per call:\n");
    printf("  Read:  sink += a[i]       1R      %6.2f GB\n", bytes / 1e9);
    printf("  Write: c[i] = 0           1W      %6.2f GB\n", bytes / 1e9);
    printf("  Copy:  c = a              1R+1W   %6.2f GB\n", 2*bytes / 1e9);
    printf("  Mul:   c = s*a            1R+1W   %6.2f GB\n", 2*bytes / 1e9);
    printf("  Add:   c = a+b            2R+1W   %6.2f GB\n", 3*bytes / 1e9);
    printf("  Triad: c = a+s*b  (FMA)   2R+1W   %6.2f GB\n", 3*bytes / 1e9);
    printf("  Dot:   sum += a*b         2R      %6.2f GB\n\n", 2*bytes / 1e9);

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMemset(d_a, 1, bytes));
    HIP_CHECK(hipMemset(d_b, 2, bytes));
    HIP_CHECK(hipMemset(d_c, 0, bytes));

    BenchEvents ev;
    const int SI = 50, SW = 5;

    // =================================================================
    //  Phase 1 — Per-operation sweep with thermal isolation
    //  Each operation sweeps all configs from a cool GPU state.
    // =================================================================

    double sweep_bw[NCONFIGS][NUM_OPS] = {};
    int    best_idx[NUM_OPS] = {};
    double best_bw [NUM_OPS] = {};

    printf("Phase 1: Sweep (50 iters/config, 3s cooldown between operations)\n\n");

    for (int op = 0; op < NUM_OPS; op++) {
        printf("  Sweeping %-6s ...", ops[op].name);
        fflush(stdout);
        for (int ci = 0; ci < NCONFIGS; ci++) {
            auto& cfg = configs[ci];
            int grid = N / (cfg.tb * 4 * cfg.mlp);
            if (grid < 1) continue;
            double bw = bench_avg(ev, [&](){
                ops[op].launch(cfg, grid, d_a, d_b, d_c, scalar, N);
            }, ops[op].data_bytes, SI, SW);
            sweep_bw[ci][op] = bw;
            if (bw > best_bw[op]) { best_bw[op] = bw; best_idx[op] = ci; }
        }
        printf("  best: TB=%-4d MLP=%d  %7.1f GB/s\n",
               configs[best_idx[op]].tb, configs[best_idx[op]].mlp, best_bw[op]);
        if (op < NUM_OPS - 1) cooldown(3);
    }

    // Print combined sweep table
    printf("\n%-5s %-4s %7s %7s %7s %7s %7s %7s %7s\n",
           "TB", "MLP", "Read", "Write", "Copy", "Mul", "Add", "Triad", "Dot");
    printf("---------------------------------------------------------------\n");
    for (int ci = 0; ci < NCONFIGS; ci++) {
        printf("%-5d %-4d", configs[ci].tb, configs[ci].mlp);
        bool tag = false;
        for (int op = 0; op < NUM_OPS; op++) {
            printf(" %7.1f", sweep_bw[ci][op]);
            if (ci == best_idx[op]) tag = true;
        }
        printf("%s\n", tag ? "  *" : "");
    }

    // =================================================================
    //  Phase 2 — Validate best configs with thermal isolation
    //  5s cooldown before each operation → burst + sustained back-to-back.
    // =================================================================

    const int BI = 50,  BW_ = 10;
    const int SUI = 200, SUW = 20;

    printf("\nPhase 2: Validate (5s cooldown before each operation)\n");
    printf("  Burst = %d iters  |  Sustained = %d iters\n\n", BI, SUI);

    Stats burst_s[NUM_OPS], sust_s[NUM_OPS];

    for (int op = 0; op < NUM_OPS; op++) {
        auto& cfg = configs[best_idx[op]];
        int grid  = N / (cfg.tb * 4 * cfg.mlp);
        auto fn   = [&](){ ops[op].launch(cfg, grid, d_a, d_b, d_c, scalar, N); };

        cooldown(5);
        burst_s[op] = bench_full(ev, fn, ops[op].data_bytes, BI,  BW_);
        sust_s[op]  = bench_full(ev, fn, ops[op].data_bytes, SUI, SUW);
    }

    // References
    Stats ref_burst[2], ref_sust[2];
    {
        cooldown(5);
        ref_burst[0] = bench_full(ev, [&](){ HIP_CHECK(hipMemsetAsync(d_c, 0, bytes)); },
                                   bytes, BI, BW_);
        ref_sust[0]  = bench_full(ev, [&](){ HIP_CHECK(hipMemsetAsync(d_c, 0, bytes)); },
                                   bytes, SUI, SUW);
    }
    {
        cooldown(5);
        ref_burst[1] = bench_full(ev, [&](){ HIP_CHECK(hipMemcpyAsync(d_c, d_a, bytes, hipMemcpyDeviceToDevice)); },
                                   2*bytes, BI, BW_);
        ref_sust[1]  = bench_full(ev, [&](){ HIP_CHECK(hipMemcpyAsync(d_c, d_a, bytes, hipMemcpyDeviceToDevice)); },
                                   2*bytes, SUI, SUW);
    }

    // =================================================================
    //  Final Report
    // =================================================================

    printf("\n========================================\n");
    printf("  Final Report  (GB/s)\n");
    printf("========================================\n\n");

    printf("%-8s %-14s %7s %7s   %7s %7s %7s\n",
           "Op", "Config", "Burst", "Sust.", "p95", "max", "min");
    printf("-------------------------------------------------------------------\n");
    for (int op = 0; op < NUM_OPS; op++) {
        auto& cfg = configs[best_idx[op]];
        char cfgstr[32];
        snprintf(cfgstr, sizeof(cfgstr), "TB=%-4d MLP=%d", cfg.tb, cfg.mlp);
        printf("%-8s %-14s %7.1f %7.1f   %7.1f %7.1f %7.1f\n",
               ops[op].name, cfgstr,
               burst_s[op].avg, sust_s[op].avg,
               sust_s[op].p95, sust_s[op].mx, sust_s[op].mn);
    }

    printf("\nReferences:\n");
    const char* ref_names[] = {"memset", "memcpy"};
    for (int r = 0; r < 2; r++) {
        printf("%-8s %-14s %7.1f %7.1f   %7.1f %7.1f %7.1f\n",
               ref_names[r], "",
               ref_burst[r].avg, ref_sust[r].avg,
               ref_sust[r].p95, ref_sust[r].mx, ref_sust[r].mn);
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    printf("\nDone.\n");
    return 0;
}
