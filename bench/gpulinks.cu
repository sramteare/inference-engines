// gpulinks.cu -- bandwidth probe between GPUs 0 and 1 and the host, using a
// single N MiB buffer per GPU so it can run next to other GPU workloads.
//
// Build: /usr/local/cuda-13.5/bin/nvcc -O2 -arch=sm_120 -o gpulinks bench/gpulinks.cu
// Big copies, P2P and remote reads:   ./gpulinks --mib 1024
// Expert-cache refill (11 MiB blobs): ./gpulinks --blob 11 --total 4096 --ring 1024
// Results: bench/results/2026-11-25-links-and-expert-cache.md
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err_));                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static const int kThreads = 256;
static const int kWarmups = 2;
static const int kSamples = 10;

// Grid-stride reader: 16-byte loads, per-thread accumulator reduced to one
// value per block so no load can be optimised away.
__global__ void readSumKernel(const uint4* __restrict__ src, size_t n16,
                              unsigned long long* __restrict__ out) {
    unsigned long long acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += stride) {
        const uint4 v = src[i];
        acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    __shared__ unsigned long long red[kThreads];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = red[0];
}

struct Gpu {
    int id = 0;
    int blocks = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    uint4* buf = nullptr;                  // N MiB device buffer / ring
    uint4* src = nullptr;                  // peer-mode source buffer (ring)
    unsigned long long* out = nullptr;     // one word per block
    void* pin = nullptr;                   // pinned host buffer
};

static void report(const char* name, const double* rates) {
    double r[kSamples];
    std::memcpy(r, rates, sizeof(r));
    std::sort(r, r + kSamples);
    const double med = 0.5 * (r[kSamples / 2 - 1] + r[kSamples / 2]);
    std::printf("%-26s median %8.2f GB/s   best %8.2f GB/s\n", name, med,
                r[kSamples - 1]);
}

// Time a 61.19 GiB expert-cache refill would take at the given rate (GB/s).
static const double kRefillBytes = 61.19 * 1024.0 * 1024.0 * 1024.0;

static void blobReport(const char* name, const double* rates, double divisor) {
    double r[kSamples];
    std::memcpy(r, rates, sizeof(r));
    std::sort(r, r + kSamples);
    const double med = 0.5 * (r[kSamples / 2 - 1] + r[kSamples / 2]);
    const double sec = kRefillBytes / ((med / divisor) * 1e9);
    if (divisor > 1.0) {
        std::printf("%-26s median %8.2f GB/s   best %8.2f GB/s   "
                    "refill 61.19 GiB/card at half total rate: %.1f s\n",
                    name, med, r[kSamples - 1], sec);
    } else {
        std::printf("%-26s median %8.2f GB/s   best %8.2f GB/s   "
                    "refill 61.19 GiB: %.1f s\n",
                    name, med, r[kSamples - 1], sec);
    }
}

// Timed loop on one device's stream (current device must be g.id).
// refillDivisor > 0 switches the output to the blob line format.
static void runSolo(const char* name, size_t bytes, const Gpu& g,
                    const std::function<void(cudaStream_t)>& op,
                    double refillDivisor = 0.0) {
    for (int i = 0; i < kWarmups; ++i) {
        op(g.stream);
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }
    double rates[kSamples];
    for (int i = 0; i < kSamples; ++i) {
        CUDA_CHECK(cudaEventRecord(g.evStart, g.stream));
        op(g.stream);
        CUDA_CHECK(cudaEventRecord(g.evStop, g.stream));
        CUDA_CHECK(cudaEventSynchronize(g.evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, g.evStart, g.evStop));
        rates[i] = (double)bytes / ((double)ms * 1e6);
    }
    if (refillDivisor > 0.0) blobReport(name, rates, refillDivisor);
    else report(name, rates);
}

// Same-size op on both GPUs at once; total bytes over the pair's wall time.
// Uses host wall clock, NOT each stream's own event interval: a stream's
// events only bracket that op once it starts, so if one GPU's op is queued
// behind other work and starts after the other has already finished, both
// event intervals are short while the pair really took twice as long -- the
// max of the two intervals would then report double the true bandwidth. A
// host span from "both GPUs idle" to "both streams done" cannot miss that.
static void runDuo(const char* name, size_t bytesTotal, const Gpu& a,
                   const Gpu& b, const std::function<void(cudaStream_t)>& opA,
                   const std::function<void(cudaStream_t)>& opB,
                   double refillDivisor = 0.0) {
    using clock = std::chrono::steady_clock;
    for (int i = 0; i < kWarmups; ++i) {
        CUDA_CHECK(cudaSetDevice(a.id));
        opA(a.stream);
        CUDA_CHECK(cudaSetDevice(b.id));
        opB(b.stream);
        CUDA_CHECK(cudaStreamSynchronize(a.stream));
        CUDA_CHECK(cudaStreamSynchronize(b.stream));
    }
    double rates[kSamples];
    for (int i = 0; i < kSamples; ++i) {
        CUDA_CHECK(cudaSetDevice(a.id));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaSetDevice(b.id));
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = clock::now();
        CUDA_CHECK(cudaSetDevice(a.id));
        opA(a.stream);
        CUDA_CHECK(cudaSetDevice(b.id));
        opB(b.stream);
        CUDA_CHECK(cudaStreamSynchronize(a.stream));
        CUDA_CHECK(cudaStreamSynchronize(b.stream));
        const auto t1 = clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        rates[i] = (double)bytesTotal / (ms * 1e6);
    }
    if (refillDivisor > 0.0) blobReport(name, rates, refillDivisor);
    else report(name, rates);
}

// ---------------------------------------------------------------------------
// Blob (expert-cache refill) mode: T/B copies of B MiB per timed run, walking
// round a device ring so the destination fits in R MiB.

struct BlobCfg {
    size_t blobBytes = 0;
    size_t ringBytes = 0;
    size_t totalBytes = 0;
    size_t nCopies = 0;
};

// Next walking offset: keeps advancing by b, but wraps to 0 before a copy
// would run past the end of the span (needs b <= span).
static size_t wrapNext(size_t off, size_t b, size_t span) {
    const size_t next = off + b;
    return (next + b > span) ? 0 : next;
}

static void enqueueBlobH2d(const Gpu& g, const BlobCfg& c, cudaStream_t st) {
    size_t d = 0, s = 0;
    for (size_t k = 0; k < c.nCopies; ++k) {
        CUDA_CHECK(cudaMemcpyAsync((char*)g.buf + d, (const char*)g.pin + s,
                                   c.blobBytes, cudaMemcpyHostToDevice, st));
        d = wrapNext(d, c.blobBytes, c.ringBytes);
        s = wrapNext(s, c.blobBytes, c.totalBytes);
    }
}

static const int kStreams4 = 4;

static void enqueueBlobH2d4(const Gpu& g, const BlobCfg& c, cudaStream_t* st) {
    size_t d = 0, s = 0;
    for (size_t k = 0; k < c.nCopies; ++k) {
        CUDA_CHECK(cudaMemcpyAsync((char*)g.buf + d, (const char*)g.pin + s,
                                   c.blobBytes, cudaMemcpyHostToDevice,
                                   st[k % kStreams4]));
        d = wrapNext(d, c.blobBytes, c.ringBytes);
        s = wrapNext(s, c.blobBytes, c.totalBytes);
    }
}

static void enqueueBlobPeer(const Gpu& src, const Gpu& dst, const BlobCfg& c,
                            cudaStream_t st) {
    size_t s = 0, d = 0;
    for (size_t k = 0; k < c.nCopies; ++k) {
        CUDA_CHECK(cudaMemcpyPeerAsync((char*)dst.buf + d, dst.id,
                                       (const char*)src.src + s, src.id,
                                       c.blobBytes, st));
        s = wrapNext(s, c.blobBytes, c.ringBytes);
        d = wrapNext(d, c.blobBytes, c.ringBytes);
    }
}

// Same op split round-robin over kStreams4 streams on one device; timed on
// the host around a sync of all four (events cannot bracket four streams).
static void runSolo4(const char* name, size_t bytes, const Gpu& g,
                     const std::function<void(cudaStream_t*)>& op) {
    using clock = std::chrono::steady_clock;
    CUDA_CHECK(cudaSetDevice(g.id));
    cudaStream_t st[kStreams4];
    for (int i = 0; i < kStreams4; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
    for (int i = 0; i < kWarmups; ++i) {
        op(st);
        for (int s = 0; s < kStreams4; ++s)
            CUDA_CHECK(cudaStreamSynchronize(st[s]));
    }
    double rates[kSamples];
    for (int i = 0; i < kSamples; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = clock::now();
        op(st);
        for (int s = 0; s < kStreams4; ++s)
            CUDA_CHECK(cudaStreamSynchronize(st[s]));
        const auto t1 = clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        rates[i] = (double)bytes / (ms * 1e6);
    }
    for (int i = 0; i < kStreams4; ++i) CUDA_CHECK(cudaStreamDestroy(st[i]));
    blobReport(name, rates, 1.0);
}

static int runBlobMode(long blob, long total, long ring) {
    if (total <= 0 || ring <= 0) {
        std::fprintf(stderr, "--total and --ring must be positive\n");
        return 1;
    }
    if (blob > ring || blob > total) {
        std::fprintf(stderr,
                     "--blob must be <= --ring and <= --total so that no "
                     "copy crosses a buffer end\n");
        return 1;
    }
    const size_t mib = 1024u * 1024u;
    BlobCfg c;
    c.blobBytes = (size_t)blob * mib;
    c.ringBytes = (size_t)ring * mib;
    c.totalBytes = (size_t)total * mib;
    c.nCopies = c.totalBytes / c.blobBytes;
    const size_t bytesPerRun = c.nCopies * c.blobBytes;
    const size_t needFree = 2 * c.ringBytes;

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        std::fprintf(stderr, "need at least 2 CUDA devices, found %d\n",
                     devCount);
        return 1;
    }

    // Gate before allocating: ring + peer source = 2 x R per GPU (pinned
    // host buffers are host memory and do not count).
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
        if (freeB < needFree) {
            std::fprintf(stderr,
                         "GPU %d has %.0f MiB free but %ld MiB (2 x %ld MiB "
                         "ring) is required; refusing to allocate\n",
                         i, (double)freeB / 1048576.0,
                         (long)(needFree / mib), ring);
            return 2;
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

    Gpu g[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc((void**)&g[i].buf, c.ringBytes));
        CUDA_CHECK(cudaMalloc((void**)&g[i].src, c.ringBytes));
        CUDA_CHECK(
            cudaStreamCreateWithFlags(&g[i].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&g[i].evStart));
        CUDA_CHECK(cudaEventCreate(&g[i].evStop));
        CUDA_CHECK(cudaMallocHost(&g[i].pin, c.totalBytes));
        std::memset(g[i].pin, 0xC0 + i, c.totalBytes);
        CUDA_CHECK(cudaMemsetAsync(g[i].src, 0x5A + i, c.ringBytes,
                                   g[i].stream));
        CUDA_CHECK(cudaStreamSynchronize(g[i].stream));
    }
    std::printf("blob mode: %ld MiB blobs x %zu copies = %zu MiB per run, "
                "ring %ld MiB\n",
                blob, c.nCopies, bytesPerRun / mib, ring);

    CUDA_CHECK(cudaSetDevice(0));
    runSolo("blob h2d gpu0", bytesPerRun, g[0],
            [&](cudaStream_t st) { enqueueBlobH2d(g[0], c, st); }, 1.0);
    CUDA_CHECK(cudaSetDevice(1));
    runSolo("blob h2d gpu1", bytesPerRun, g[1],
            [&](cudaStream_t st) { enqueueBlobH2d(g[1], c, st); }, 1.0);
    runSolo4("blob h2d gpu1 4 streams", bytesPerRun, g[1],
             [&](cudaStream_t* st) { enqueueBlobH2d4(g[1], c, st); });
    runDuo("blob h2d both at once", 2 * bytesPerRun, g[0], g[1],
           [&](cudaStream_t st) { enqueueBlobH2d(g[0], c, st); },
           [&](cudaStream_t st) { enqueueBlobH2d(g[1], c, st); }, 2.0);
    CUDA_CHECK(cudaSetDevice(0));
    runSolo("blob peer 0->1", bytesPerRun, g[0],
            [&](cudaStream_t st) { enqueueBlobPeer(g[0], g[1], c, st); }, 1.0);
    CUDA_CHECK(cudaSetDevice(1));
    runSolo("blob peer 1->0", bytesPerRun, g[1],
            [&](cudaStream_t st) { enqueueBlobPeer(g[1], g[0], c, st); }, 1.0);

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(g[i].buf));
        CUDA_CHECK(cudaFree(g[i].src));
        CUDA_CHECK(cudaFreeHost(g[i].pin));
        CUDA_CHECK(cudaStreamDestroy(g[i].stream));
        CUDA_CHECK(cudaEventDestroy(g[i].evStart));
        CUDA_CHECK(cudaEventDestroy(g[i].evStop));
    }
    std::printf("done\n");
    return 0;
}

int main(int argc, char** argv) {
    long mib = 512;
    long blob = 0, total = 4096, ring = 1024;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            mib = std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--blob") == 0 && i + 1 < argc) {
            blob = std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--total") == 0 && i + 1 < argc) {
            total = std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--ring") == 0 && i + 1 < argc) {
            ring = std::strtol(argv[++i], nullptr, 10);
        } else {
            std::fprintf(
                stderr,
                "usage: %s [--mib N] | --blob B [--total T] [--ring R]\n",
                argv[0]);
            return 1;
        }
    }
    if (blob > 0) return runBlobMode(blob, total, ring);
    if (mib <= 0) {
        std::fprintf(stderr, "--mib must be a positive number of MiB\n");
        return 1;
    }
    const size_t bytes = (size_t)mib * 1024u * 1024u;
    const size_t needFree = (size_t)3 * bytes;
    std::printf(
        "buffer: %ld MiB per GPU; local-read and host-copy numbers are only "
        "meaningful when the buffer is bigger than the GPU L2 (~128 MB) and "
        "the CPU L3 (~64 MB), so use --mib 512 or more for those\n",
        mib);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        std::fprintf(stderr, "need at least 2 CUDA devices, found %d\n",
                     devCount);
        return 1;
    }

    // Gate on free memory of BOTH GPUs before allocating anything anywhere.
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
        if (freeB < needFree) {
            std::fprintf(stderr,
                         "GPU %d has %.0f MiB free but %ld MiB (3 x %ld MiB) "
                         "is required; refusing to allocate\n",
                         i, (double)freeB / 1048576.0,
                         (long)(needFree / 1048576), mib);
            return 2;
        }
    }

    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    std::printf(
        "p2p-access: canAccessPeer 0->1 %s, 1->0 %s (enabled both ways)\n",
        can01 ? "yes" : "no", can10 ? "yes" : "no");

    Gpu g[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        cudaDeviceProp dp;
        CUDA_CHECK(cudaGetDeviceProperties(&dp, i));
        if (i == 0) {
            std::printf("gpu0: %s, %d SMs\n", dp.name, dp.multiProcessorCount);
        }
        g[i].id = i;
        g[i].blocks = dp.multiProcessorCount * 4;
        CUDA_CHECK(cudaMalloc((void**)&g[i].buf, bytes));
        CUDA_CHECK(cudaMalloc((void**)&g[i].out,
                              (size_t)g[i].blocks *
                                  sizeof(unsigned long long)));
        CUDA_CHECK(
            cudaStreamCreateWithFlags(&g[i].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&g[i].evStart));
        CUDA_CHECK(cudaEventCreate(&g[i].evStop));
        CUDA_CHECK(cudaMallocHost(&g[i].pin, bytes));
        std::memset(g[i].pin, 0xC0 + i, bytes);
        CUDA_CHECK(cudaMemsetAsync(g[i].buf, 0xA5 + i, bytes, g[i].stream));
        CUDA_CHECK(cudaStreamSynchronize(g[i].stream));
    }

    char name[64];

    for (int s = 0; s < 2; ++s) {
        const int d = 1 - s;
        std::snprintf(name, sizeof(name), "copy %d->%d", s, d);
        CUDA_CHECK(cudaSetDevice(s));
        runSolo(name, bytes, g[s], [&](cudaStream_t st) {
            CUDA_CHECK(cudaMemcpyPeerAsync(g[d].buf, d, g[s].buf, s, bytes,
                                           st));
        });
    }

    runDuo("copy 0<->1 both at once", 2 * bytes, g[0], g[1],
           [&](cudaStream_t st) {
               CUDA_CHECK(cudaMemcpyPeerAsync(g[1].buf, 1, g[0].buf, 0, bytes,
                                              st));
           },
           [&](cudaStream_t st) {
               CUDA_CHECK(cudaMemcpyPeerAsync(g[0].buf, 0, g[1].buf, 1, bytes,
                                              st));
           });

    for (int r = 0; r < 2; ++r) {
        const int o = 1 - r;
        std::snprintf(name, sizeof(name), "read remote %d reads %d", r, o);
        CUDA_CHECK(cudaSetDevice(r));
        const uint4* src = g[o].buf;
        const int nb = g[r].blocks;
        runSolo(name, bytes, g[r], [&](cudaStream_t st) {
            readSumKernel<<<nb, kThreads, 0, st>>>(src, bytes / 16, g[r].out);
            CUDA_CHECK(cudaGetLastError());
        });
    }
    for (int r = 0; r < 2; ++r) {
        std::snprintf(name, sizeof(name), "read local %d", r);
        CUDA_CHECK(cudaSetDevice(r));
        const uint4* src = g[r].buf;
        const int nb = g[r].blocks;
        runSolo(name, bytes, g[r], [&](cudaStream_t st) {
            readSumKernel<<<nb, kThreads, 0, st>>>(src, bytes / 16, g[r].out);
            CUDA_CHECK(cudaGetLastError());
        });
    }

    for (int i = 0; i < 2; ++i) {
        std::snprintf(name, sizeof(name), "h2d gpu%d", i);
        CUDA_CHECK(cudaSetDevice(i));
        runSolo(name, bytes, g[i], [&](cudaStream_t st) {
            CUDA_CHECK(cudaMemcpyAsync(g[i].buf, g[i].pin, bytes,
                                       cudaMemcpyHostToDevice, st));
        });
    }

    runDuo("h2d both at once", 2 * bytes, g[0], g[1],
           [&](cudaStream_t st) {
               CUDA_CHECK(cudaMemcpyAsync(g[0].buf, g[0].pin, bytes,
                                          cudaMemcpyHostToDevice, st));
           },
           [&](cudaStream_t st) {
               CUDA_CHECK(cudaMemcpyAsync(g[1].buf, g[1].pin, bytes,
                                          cudaMemcpyHostToDevice, st));
           });

    for (int i = 0; i < 2; ++i) {
        std::snprintf(name, sizeof(name), "d2h gpu%d", i);
        CUDA_CHECK(cudaSetDevice(i));
        runSolo(name, bytes, g[i], [&](cudaStream_t st) {
            CUDA_CHECK(cudaMemcpyAsync(g[i].pin, g[i].buf, bytes,
                                       cudaMemcpyDeviceToHost, st));
        });
    }

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(g[i].buf));
        CUDA_CHECK(cudaFree(g[i].out));
        CUDA_CHECK(cudaFreeHost(g[i].pin));
        CUDA_CHECK(cudaStreamDestroy(g[i].stream));
        CUDA_CHECK(cudaEventDestroy(g[i].evStart));
        CUDA_CHECK(cudaEventDestroy(g[i].evStop));
    }
    std::printf("done\n");
    return 0;
}
