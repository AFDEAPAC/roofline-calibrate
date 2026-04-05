# roofline-calibrate

ISA-optimised GPU memory bandwidth benchmark for AMD Instinct GPUs.

Uses 128-bit (`v4f`) non-temporal loads/stores and automatically sweeps
Thread Block Size (TB) × Memory-Level Parallelism (MLP) to find peak
bandwidth for each operation on any platform.

## Operations

| Operation | Formula | Data moved |
|-----------|---------|------------|
| Read | `sink += a[i]` | 1R = 1× array |
| Write | `c[i] = 0` | 1W = 1× array |
| Copy | `c[i] = a[i]` | 1R+1W = 2× array |
| Mul | `c[i] = scalar * a[i]` | 1R+1W = 2× array |
| Add | `c[i] = a[i] + b[i]` | 2R+1W = 3× array |
| Triad | `c[i] = a[i] + scalar*b[i]` | 2R+1W = 3× array |
| Dot | `sum += a[i] * b[i]` | 2R = 2× array |

Plus reference measurements using `hipMemsetAsync` and `hipMemcpyAsync`.

## Build

Requires ROCm / HIP toolchain (`hipcc`).

```bash
make                                    # default: gfx942 (MI300X/MI308X)
make ARCHFLAG=--offload-arch=gfx90a     # MI210/MI250X
make ARCHFLAG=--offload-arch=gfx1100    # RDNA3
```

## Run

```bash
HIP_FORCE_DEV_KERNARG=1 ./build/stream_ops
```

Custom array size (number of floats, default auto-detects ≥ 2 GB):

```bash
HIP_FORCE_DEV_KERNARG=1 ./build/stream_ops 671088640
```

## How it works

1. **Sweep** — Each operation is swept independently across 20 configs
   (TB ∈ {64, 128, 256, 512, 1024} × MLP ∈ {1, 2, 4, 8}), with a 3-second
   GPU cooldown between operations to eliminate thermal cross-contamination.

2. **Validate** — The best config per operation is re-measured with a
   5-second cooldown beforehand:
   - **Burst**: 50 iterations (captures peak before any thermal throttle)
   - **Sustained**: 200 iterations (represents continuous workload)

3. **Report** — Final table with burst, sustained, p95, max, min for all
   7 operations plus `hipMemsetAsync` / `hipMemcpyAsync` references.

## Example output

```
========================================
  Final Report  (GB/s)
========================================

Op       Config           Burst   Sust.       p95     max     min
-------------------------------------------------------------------
Read     TB=128  MLP=2   3930.0  3967.8    4026.3  4045.5  3663.4
Write    TB=512  MLP=2   2827.8  2829.2    2846.0  2850.0  2724.6
Copy     TB=64   MLP=2   3565.9  3581.3    3588.6  3591.7  3519.3
Mul      TB=64   MLP=2   3494.6  3528.8    3540.6  3545.2  3450.1
Add      TB=64   MLP=1   3795.7  3803.1    3864.1  3912.5  3715.4
Triad    TB=64   MLP=1   3770.9  3759.6    3822.1  3843.3  3667.6
Dot      TB=64   MLP=2   3749.0  3719.9    3831.9  3888.9  3534.8

References:
memset                   2800.5  2803.7    2814.4  2818.3  2723.1
memcpy                   1828.5  1829.6    1835.1  1837.8  1821.5
```

*(AMD Instinct MI308X, ROCm 7.2, array = 2.15 GB)*

## Project structure

```
roofline-calibrate/
├── Makefile                 # Auto-discovers benchmarks/*.hip
├── include/
│   ├── bench_common.h       # Shared: HIP_CHECK, Stats, bench helpers, GPU info
│   └── types.h              # v4f, v2f, v2d, v4d vector types
└── benchmarks/
    └── stream_ops.hip       # Full 7-operation benchmark
```

To add a new benchmark, drop a `.hip` file into `benchmarks/` and run `make`.
