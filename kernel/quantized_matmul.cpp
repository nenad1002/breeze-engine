// int4_gemm.cpp — MatMulNBits (4-bit, block_size=32) CPU kernel using AVX-512
// VNNI int8 dot-products (VPDPBUSD). Quantizing activations to int8 lets the
// int8 dot skip the per-weight int->fp32 conversion and do 4 MACs per lane, so
// compute stops being the bottleneck and the kernel becomes DRAM-bandwidth bound.
//
// Math per block (32 k-values), per output column n, per row m:
//   B int8:  b = (nibble - zp[n,blk])            in [-15,15]
//   A int8:  per-(row,block) symmetric quant a_s8 = round(A/a_scale); a_u8 = a_s8+128
//   VPDPBUSD accumulates sum(a_u8 * b) in int32; correct the +128 bias with
//   corr = 128*sum_k(b):   sum(a_s8*b) = acc_i32 - corr
//   C[m,n] += a_scale[m,blk] * b_scale[n,blk] * (acc_i32 - corr)
//
// Weight prepack layout (per column-tile of 16, per block, per kgroup of 4 k):
//   Bv[32 bytes] = 64 int4 = v[col*4 + k], packed 2/byte, first 32 vals in
//   bytes[0:16], next 32 in bytes[16:32] (so each 128-bit half unpacks locally).

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <omp.h>
#include <sched.h>

namespace {

// CPU IDs at or above this configured split use the optional second-node replica.
// Overridable via i4_set_numa_split(). 1<<30 disables replication use.
int g_node1_cpu_start = 48;

// Hi-precision activation: 0 = int8 activation (fast, ~8-bit); 1 = int16 2-pass
// (accurate, ~15-bit). Set via i4_set_hi_prec. Weights are identical either way.
int g_hi_prec = 0;

struct PackedWeight {
    int N, K, nblk, ntiles;
    int bits;         // 4 or 8; selects int4 (nibble) vs int8 (byte) B layout
    uint8_t* B;       // int4: [nt,nblk,8,32] nibble; int8: [nt,nblk,8,64] signed(w-128)
    float* scales;    // [ntiles, nblk, 16]
    int8_t* zp;       // [ntiles, nblk, 16]  (int4 only; nullptr for int8)
    int32_t* corr;    // [ntiles, nblk, 16]  = 128 * sum_k(w - zp)
    // Optional node-1 replica (first-touched on NUMA node 1). Same layout.
    uint8_t* B1;
    float* scales1;
    int8_t* zp1;
    int32_t* corr1;
};

inline void* aa64(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - 63) return nullptr;
    void* p = nullptr;
    if (posix_memalign(&p, 64, (bytes + 63) & ~size_t(63)) != 0) return nullptr;
    return p;
}

void free_packed(PackedWeight* P) {
    if (!P) return;
    std::free(P->B); std::free(P->scales); std::free(P->zp); std::free(P->corr);
    std::free(P->B1); std::free(P->scales1); std::free(P->zp1); std::free(P->corr1);
    delete P;
}

PackedWeight* allocate_packed(int N, int K, int nblk, int bits) {
    if (N <= 0 || K <= 0 || N % 16 || K % 32 || nblk != K / 32 ||
        (bits != 4 && bits != 8)) return nullptr;
    if (size_t(N) > (std::numeric_limits<size_t>::max() - 63) /
                       size_t(nblk) / size_t(4 * bits)) return nullptr;
    auto* P = new (std::nothrow) PackedWeight{};
    if (!P) return nullptr;
    P->N = N; P->K = K; P->nblk = nblk; P->ntiles = N / 16; P->bits = bits;
    const size_t blocks = size_t(N) * nblk;
    const size_t weight_bytes = blocks * (4 * bits);
    P->B = static_cast<uint8_t*>(aa64(weight_bytes));
    P->scales = static_cast<float*>(aa64(blocks * sizeof(float)));
    P->zp = bits == 4 ? static_cast<int8_t*>(aa64(blocks)) : nullptr;
    P->corr = static_cast<int32_t*>(aa64(blocks * sizeof(int32_t)));
    if (!P->B || !P->scales || !P->corr || (bits == 4 && !P->zp)) {
        free_packed(P);
        return nullptr;
    }

    // Retain the existing replication policy. The primary packing loop uses
    // the same static tile schedule as inference for NUMA first-touch locality.
    bool replicate = false;
    #pragma omp parallel
    {
        #pragma omp single
        replicate = (omp_get_num_threads() > g_node1_cpu_start) &&
                    (g_node1_cpu_start < (1 << 20));
    }
    if (replicate) {
        P->B1 = static_cast<uint8_t*>(aa64(weight_bytes));
        P->scales1 = static_cast<float*>(aa64(blocks * sizeof(float)));
        P->zp1 = bits == 4 ? static_cast<int8_t*>(aa64(blocks)) : nullptr;
        P->corr1 = static_cast<int32_t*>(aa64(blocks * sizeof(int32_t)));
        if (!P->B1 || !P->scales1 || !P->corr1 || (bits == 4 && !P->zp1)) {
            free_packed(P);
            return nullptr;
        }
    }
    return P;
}

void populate_replica(PackedWeight* P) {
    if (!P->B1) return;
    const size_t bstride = size_t(P->nblk) * 8 * (P->bits == 8 ? 64 : 32);
    const size_t sstride = size_t(P->nblk) * 16;
    auto copy_tile = [&](int t) {
        const size_t b = size_t(t) * bstride, s = size_t(t) * sstride;
        std::memcpy(P->B1 + b, P->B + b, bstride);
        std::memcpy(P->scales1 + s, P->scales + s, sstride * sizeof(float));
        if (P->zp1) std::memcpy(P->zp1 + s, P->zp + s, sstride);
        std::memcpy(P->corr1 + s, P->corr + s, sstride * sizeof(int32_t));
    };
    int workers = 0;
    #pragma omp parallel
    {
        const bool node1 = sched_getcpu() >= g_node1_cpu_start;
        int rank = -1;
        if (node1) {
            #pragma omp atomic capture
            rank = workers++;
        }
        #pragma omp barrier
        if (node1) {
            // Initialize EVERY tile, not only this thread's inference slice:
            // later calls can use a different thread count or affinity.
            for (int t = rank; t < P->ntiles; t += workers) copy_tile(t);
        }
        #pragma omp single
        {
            // A user-specified CPU split may not intersect the actual affinity.
            if (!workers) for (int t = 0; t < P->ntiles; ++t) copy_tile(t);
        }
    }
}

template <int Bits, bool HalfScales>
void pack_onnx(PackedWeight* P, const uint8_t* qw, const void* scales,
               const uint8_t* qzeros) {
    const int nblk = P->nblk;
    constexpr int block_bytes = 4 * Bits;
    constexpr int group_bytes = Bits == 8 ? 64 : 32;
    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < P->ntiles; ++tile) {
        for (int kb = 0; kb < nblk; ++kb) {
            const size_t out_block = size_t(tile) * nblk + kb;
            uint8_t* dest = P->B + out_block * 8 * group_bytes;
            for (int col = 0; col < 16; ++col) {
                const size_t row = size_t(tile) * 16 + col;
                const size_t in_block = row * nblk + kb;
                const size_t meta = out_block * 16 + col;
                const uint8_t* src = qw + in_block * block_bytes;
                if constexpr (HalfScales)
                    P->scales[meta] = _cvtsh_ss(static_cast<const uint16_t*>(scales)[in_block]);
                else
                    P->scales[meta] = static_cast<const float*>(scales)[in_block];

                int sum, zp;
                if constexpr (Bits == 8) {
                    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
                    const __m256i sums = _mm256_sad_epu8(raw, _mm256_setzero_si256());
                    const __m128i halves = _mm_add_epi64(_mm256_castsi256_si128(sums),
                                                       _mm256_extracti128_si256(sums, 1));
                    sum = int(_mm_cvtsi128_si64(halves) + _mm_extract_epi64(halves, 1));
                    zp = 128;
                    for (int group = 0; group < 8; ++group) {
                        uint32_t word;
                        std::memcpy(&word, src + group * 4, sizeof(word));
                        word ^= 0x80808080u;  // uint8 offset-128 -> signed int8 bytes
                        std::memcpy(dest + group * 64 + col * 4, &word, sizeof(word));
                    }
                } else {
                    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
                    const __m128i mask = _mm_set1_epi8(15);
                    const __m128i pairs = _mm_add_epi8(_mm_and_si128(raw, mask),
                                                     _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
                    const __m128i sums = _mm_sad_epu8(pairs, _mm_setzero_si128());
                    sum = int(_mm_cvtsi128_si64(sums) + _mm_extract_epi64(sums, 1));
                    zp = qzeros ? (qzeros[row * ((nblk + 1) / 2) + kb / 2] >> (4 * (kb % 2))) & 15 : 8;
                    P->zp[meta] = int8_t(zp);
                    for (int group = 0; group < 8; ++group) {
                        // Four adjacent K values already occupy two packed bytes.
                        // Reorder the bytes directly; never expand a weight tensor.
                        std::memcpy(dest + group * 32 + col * 2, src + group * 2, 2);
                    }
                }
                P->corr[meta] = 128 * (sum - 32 * zp);
            }
        }
    }
}

// Indices to expand zp[16] -> zp[64] with each column's zp repeated 4x (i -> i/4).
inline __m512i zp_expand_idx() {
    alignas(64) uint8_t idx[64];
    for (int i = 0; i < 64; ++i) idx[i] = (uint8_t)(i / 4);
    return _mm512_load_si512(reinterpret_cast<const void*>(idx));
}

// 32 bytes (64 int4) -> 64 int8 in order v[0..63] = [col0:k0..3, col1:k0..3, ...].
inline __m512i unpack_kgroup(const uint8_t* p) {
    const __m128i lm = _mm_set1_epi8(0x0F);
    __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));       // v0..31
    __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));  // v32..63
    __m128i lo0 = _mm_and_si128(x0, lm), hi0 = _mm_and_si128(_mm_srli_epi16(x0, 4), lm);
    __m256i v031 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
    __m128i lo1 = _mm_and_si128(x1, lm), hi1 = _mm_and_si128(_mm_srli_epi16(x1, 4), lm);
    __m256i v3263 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(v031), v3263, 1);
}

void quantize_A(const float* A, int M, int K, int nblk, uint8_t* au8, float* ascale) {
    for (int m = 0; m < M; ++m) {
        for (int kb = 0; kb < nblk; ++kb) {
            const float* a = A + (size_t)m * K + kb * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) { float v = std::fabs(a[i]); if (v > amax) amax = v; }
            float scale = amax / 127.0f;
            float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            ascale[(size_t)m * nblk + kb] = scale;
            uint8_t* o = au8 + (size_t)m * K + kb * 32;
            for (int i = 0; i < 32; ++i) {
                int q = (int)std::lrint(a[i] * inv);
                q = q > 127 ? 127 : (q < -127 ? -127 : q);
                o[i] = (uint8_t)(q + 128);
            }
        }
    }
}

// Hi-precision activation quant: per-block symmetric int16, split into two u8
// bytes (hi, lo) so we can use two dpbusd passes. Recovers ~15-bit activation
// precision (int8 is too lossy for deep/recurrent models).
void quantize_A_i16(const float* A, int M, int K, int nblk,
                    uint8_t* au8_hi, uint8_t* au8_lo, float* ascale) {
    for (int m = 0; m < M; ++m) {
        for (int kb = 0; kb < nblk; ++kb) {
            const float* a = A + (size_t)m * K + kb * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) { float v = std::fabs(a[i]); if (v > amax) amax = v; }
            float scale = amax / 32767.0f;
            float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            ascale[(size_t)m * nblk + kb] = scale;
            uint8_t* hi = au8_hi + (size_t)m * K + kb * 32;
            uint8_t* lo = au8_lo + (size_t)m * K + kb * 32;
            for (int i = 0; i < 32; ++i) {
                int q = (int)std::lrint(a[i] * inv);
                q = q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
                unsigned u = (unsigned)(q + 32768);
                hi[i] = (uint8_t)(u >> 8);
                lo[i] = (uint8_t)(u & 0xFF);
            }
        }
    }
}

// HP combine: cs = 256*(acc_hi - corr) + acc_lo  (== sum(a_s16*b) since
// 32768*sum(b) = 256*corr).  Non-HP: cs = acc_hi - corr.
template <bool HP>
inline __m512i hp_combine(__m512i acc_hi, __m512i acc_lo, __m512i corr16) {
    if constexpr (HP)
        return _mm512_add_epi32(_mm512_slli_epi32(_mm512_sub_epi32(acc_hi, corr16), 8), acc_lo);
    else
        return _mm512_sub_epi32(acc_hi, corr16);
}

template <int M, bool HP>
inline void tile_kernel(const PackedWeight* P, const uint8_t* Bbase, const float* scbase,
                        const int8_t* zpbase, const int32_t* corrbase,
                        const uint8_t* au8, const uint8_t* au8_lo, const float* ascale,
                        int t, float* C) {
    const int N = P->N, K = P->K, nblk = P->nblk;
    const uint8_t* Bv = Bbase + (size_t)t * nblk * 8 * 32;
    const float* sc = scbase + (size_t)t * nblk * 16;
    const int8_t* zpp = zpbase + (size_t)t * nblk * 16;
    const int32_t* corrp = corrbase + (size_t)t * nblk * 16;
    const __m512i ZPIDX = zp_expand_idx();

    __m512 acc[M];
    for (int m = 0; m < M; ++m) acc[m] = _mm512_setzero_ps();

    for (int kb = 0; kb < nblk; ++kb) {
        const uint8_t* bblk = Bv + (size_t)kb * 8 * 32;
        // prefetch 4 blocks ahead (~4 KB) to hide DRAM latency, not just 1
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 32 + 128), _MM_HINT_T0);
        const __m512 bscale = _mm512_loadu_ps(sc + kb * 16);
        const __m512i corr16 = _mm512_loadu_si512(reinterpret_cast<const void*>(corrp + kb * 16));
        __m128i zp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(zpp + kb * 16));
        __m512i zp64 = _mm512_permutexvar_epi8(ZPIDX, _mm512_castsi128_si512(zp16));

        __m512i acci[M], accl[M];
        for (int m = 0; m < M; ++m) { acci[m] = _mm512_setzero_si512(); accl[m] = _mm512_setzero_si512(); }

        for (int g = 0; g < 8; ++g) {
            __m512i b = _mm512_sub_epi8(unpack_kgroup(bblk + g * 32), zp64);
            for (int m = 0; m < M; ++m) {
                __m512i a = _mm512_set1_epi32(
                    *reinterpret_cast<const int32_t*>(au8 + (size_t)m * K + kb * 32 + g * 4));
                acci[m] = _mm512_dpbusd_epi32(acci[m], a, b);
                if constexpr (HP) {
                    __m512i al = _mm512_set1_epi32(
                        *reinterpret_cast<const int32_t*>(au8_lo + (size_t)m * K + kb * 32 + g * 4));
                    accl[m] = _mm512_dpbusd_epi32(accl[m], al, b);
                }
            }
        }
        for (int m = 0; m < M; ++m) {
            __m512i cs = hp_combine<HP>(acci[m], accl[m], corr16);
            __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(cs), bscale);
            acc[m] = _mm512_fmadd_ps(_mm512_set1_ps(ascale[(size_t)m * nblk + kb]), f, acc[m]);
        }
    }
    for (int m = 0; m < M; ++m) _mm512_storeu_ps(C + (size_t)m * N + t * 16, acc[m]);
}

template <int M, bool HP>
inline void tile_kernel8(const PackedWeight* P, const int8_t* Bbase, const float* scbase,
                         const int32_t* corrbase, const uint8_t* au8, const uint8_t* au8_lo,
                         const float* ascale, int t, float* C) {
    const int N = P->N, K = P->K, nblk = P->nblk;
    const int8_t* Bv = Bbase + (size_t)t * nblk * 8 * 64;
    const float* sc = scbase + (size_t)t * nblk * 16;
    const int32_t* corrp = corrbase + (size_t)t * nblk * 16;

    __m512 acc[M];
    for (int m = 0; m < M; ++m) acc[m] = _mm512_setzero_ps();

    for (int kb = 0; kb < nblk; ++kb) {
        const int8_t* bblk = Bv + (size_t)kb * 8 * 64;
        // prefetch 4 blocks ahead (~8 KB) to hide DRAM latency, not just 1
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 64 + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 64 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(bblk + 4 * 8 * 64 + 384), _MM_HINT_T0);
        const __m512 bscale = _mm512_loadu_ps(sc + kb * 16);
        const __m512i corr16 = _mm512_loadu_si512(reinterpret_cast<const void*>(corrp + kb * 16));

        __m512i acci[M], accl[M];
        for (int m = 0; m < M; ++m) { acci[m] = _mm512_setzero_si512(); accl[m] = _mm512_setzero_si512(); }

        for (int g = 0; g < 8; ++g) {
            __m512i b = _mm512_loadu_si512(reinterpret_cast<const void*>(bblk + g * 64));
            for (int m = 0; m < M; ++m) {
                __m512i a = _mm512_set1_epi32(
                    *reinterpret_cast<const int32_t*>(au8 + (size_t)m * K + kb * 32 + g * 4));
                acci[m] = _mm512_dpbusd_epi32(acci[m], a, b);
                if constexpr (HP) {
                    __m512i al = _mm512_set1_epi32(
                        *reinterpret_cast<const int32_t*>(au8_lo + (size_t)m * K + kb * 32 + g * 4));
                    accl[m] = _mm512_dpbusd_epi32(accl[m], al, b);
                }
            }
        }
        for (int m = 0; m < M; ++m) {
            __m512i cs = hp_combine<HP>(acci[m], accl[m], corr16);
            __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(cs), bscale);
            acc[m] = _mm512_fmadd_ps(_mm512_set1_ps(ascale[(size_t)m * nblk + kb]), f, acc[m]);
        }
    }
    for (int m = 0; m < M; ++m) _mm512_storeu_ps(C + (size_t)m * N + t * 16, acc[m]);
}

// Tile loop assuming we are ALREADY inside an omp parallel region. No barrier
// at the end (nowait) — caller adds one when needed.
template <int M, bool HP>
inline void run_tiles_for(const PackedWeight* P, const uint8_t* au8, const uint8_t* au8_lo,
                          const float* ascale, float* C) {
    const int ntiles = P->ntiles;
    const bool repl = (P->B1 != nullptr);
    const uint8_t* Bb = P->B; const float* scb = P->scales;
    const int8_t* zpb = P->zp; const int32_t* corrb = P->corr;
    if (repl && sched_getcpu() >= g_node1_cpu_start) {
        Bb = P->B1; scb = P->scales1; zpb = P->zp1; corrb = P->corr1;
    }
    #pragma omp for schedule(static) nowait
    for (int t = 0; t < ntiles; ++t)
        tile_kernel<M, HP>(P, Bb, scb, zpb, corrb, au8, au8_lo, ascale, t, C);
}

template <int M, bool HP>
void run_tiles(const PackedWeight* P, const uint8_t* au8, const uint8_t* au8_lo,
               const float* ascale, float* C) {
    #pragma omp parallel
    run_tiles_for<M, HP>(P, au8, au8_lo, ascale, C);
}

template <int M, bool HP>
inline void run_tiles_for8(const PackedWeight* P, const uint8_t* au8, const uint8_t* au8_lo,
                           const float* ascale, float* C) {
    const int ntiles = P->ntiles;
    const bool repl = (P->B1 != nullptr);
    const int8_t* Bb = reinterpret_cast<const int8_t*>(P->B);
    const float* scb = P->scales; const int32_t* corrb = P->corr;
    if (repl && sched_getcpu() >= g_node1_cpu_start) {
        Bb = reinterpret_cast<const int8_t*>(P->B1); scb = P->scales1; corrb = P->corr1;
    }
    #pragma omp for schedule(static) nowait
    for (int t = 0; t < ntiles; ++t)
        tile_kernel8<M, HP>(P, Bb, scb, corrb, au8, au8_lo, ascale, t, C);
}

template <int M, bool HP>
void run_tiles8(const PackedWeight* P, const uint8_t* au8, const uint8_t* au8_lo,
                const float* ascale, float* C) {
    #pragma omp parallel
    run_tiles_for8<M, HP>(P, au8, au8_lo, ascale, C);
}

// Compile-time HP dispatch over M (1..8) and weight bits (4/8).
template <bool HP>
inline void dispatch_run(const PackedWeight* P, int mm, const uint8_t* a8,
                         const uint8_t* alo, const float* asc, float* c) {
    if (P->bits == 8) {
        switch (mm) {
            case 1: run_tiles8<1, HP>(P, a8, alo, asc, c); break;
            case 2: run_tiles8<2, HP>(P, a8, alo, asc, c); break;
            case 3: run_tiles8<3, HP>(P, a8, alo, asc, c); break;
            case 4: run_tiles8<4, HP>(P, a8, alo, asc, c); break;
            case 5: run_tiles8<5, HP>(P, a8, alo, asc, c); break;
            case 6: run_tiles8<6, HP>(P, a8, alo, asc, c); break;
            case 7: run_tiles8<7, HP>(P, a8, alo, asc, c); break;
            case 8: run_tiles8<8, HP>(P, a8, alo, asc, c); break;
        }
    } else {
        switch (mm) {
            case 1: run_tiles<1, HP>(P, a8, alo, asc, c); break;
            case 2: run_tiles<2, HP>(P, a8, alo, asc, c); break;
            case 3: run_tiles<3, HP>(P, a8, alo, asc, c); break;
            case 4: run_tiles<4, HP>(P, a8, alo, asc, c); break;
            case 5: run_tiles<5, HP>(P, a8, alo, asc, c); break;
            case 6: run_tiles<6, HP>(P, a8, alo, asc, c); break;
            case 7: run_tiles<7, HP>(P, a8, alo, asc, c); break;
            case 8: run_tiles<8, HP>(P, a8, alo, asc, c); break;
        }
    }
}

template <bool HP>
inline void dispatch_run_region(const PackedWeight* P, int M, const uint8_t* a8,
                                const uint8_t* alo, const float* asc, float* c) {
    if (P->bits == 8) {
        switch (M) {
            case 1: run_tiles_for8<1, HP>(P, a8, alo, asc, c); break;
            case 2: run_tiles_for8<2, HP>(P, a8, alo, asc, c); break;
            case 3: run_tiles_for8<3, HP>(P, a8, alo, asc, c); break;
            case 4: run_tiles_for8<4, HP>(P, a8, alo, asc, c); break;
            case 5: run_tiles_for8<5, HP>(P, a8, alo, asc, c); break;
            case 6: run_tiles_for8<6, HP>(P, a8, alo, asc, c); break;
            case 7: run_tiles_for8<7, HP>(P, a8, alo, asc, c); break;
            case 8: run_tiles_for8<8, HP>(P, a8, alo, asc, c); break;
        }
    } else {
        switch (M) {
            case 1: run_tiles_for<1, HP>(P, a8, alo, asc, c); break;
            case 2: run_tiles_for<2, HP>(P, a8, alo, asc, c); break;
            case 3: run_tiles_for<3, HP>(P, a8, alo, asc, c); break;
            case 4: run_tiles_for<4, HP>(P, a8, alo, asc, c); break;
            case 5: run_tiles_for<5, HP>(P, a8, alo, asc, c); break;
            case 6: run_tiles_for<6, HP>(P, a8, alo, asc, c); break;
            case 7: run_tiles_for<7, HP>(P, a8, alo, asc, c); break;
            case 8: run_tiles_for<8, HP>(P, a8, alo, asc, c); break;
        }
    }
}

}  // namespace

extern "C" {

void i4_set_threads(int n) { omp_set_num_threads(n); }
void i4_set_numa_split(int cpu) { g_node1_cpu_start = cpu; }
void i4_set_hi_prec(int on) { g_hi_prec = on; }

// Legacy entry point: copy an already-transformed VNNI layout.
// srcB: [ntiles,nblk,8,32 or 64]; metadata: [ntiles,nblk,16].
void* i4_prepack(const uint8_t* srcB, const float* srcScale, const int8_t* srcZp,
                 const int32_t* srcCorr, int N, int K, int nblk, int bits) {
    if (!srcB || !srcScale || !srcCorr || (bits == 4 && !srcZp)) return nullptr;
    auto* P = allocate_packed(N, K, nblk, bits);
    if (!P) return nullptr;
    const int ntiles = P->ntiles;
    const int kgbytes = (bits == 8) ? 64 : 32;
    const size_t bstride = (size_t)nblk * 8 * kgbytes;
    const size_t sstride = (size_t)nblk * 16;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < ntiles; ++t) {
        std::memcpy(P->B + (size_t)t * bstride, srcB + (size_t)t * bstride, bstride);
        std::memcpy(P->scales + (size_t)t * sstride, srcScale + (size_t)t * sstride, sstride * sizeof(float));
        if (P->zp) std::memcpy(P->zp + (size_t)t * sstride, srcZp + (size_t)t * sstride, sstride);
        std::memcpy(P->corr + (size_t)t * sstride, srcCorr + (size_t)t * sstride, sstride * sizeof(int32_t));
    }
    populate_replica(P);
    return P;
}

// Fused ONNX -> VNNI packing. Inputs are contiguous [N,K/32,16 or 32]
// raw bytes, [N,K/32] FP16/FP32 scales, and optional packed INT4 zero points.
// scale_bits is 16 or 32. INT8 zero points must have been validated as 128
// by the caller; pass nullptr for them. Output owns all of its storage.
void* i4_prepack_onnx(const uint8_t* qw, const void* scales, const uint8_t* qzeros,
                      int N, int K, int bits, int scale_bits) {
    if (!qw || !scales || (scale_bits != 16 && scale_bits != 32) ||
        (bits == 8 && qzeros)) return nullptr;
    auto* P = allocate_packed(N, K, K / 32, bits);
    if (!P) return nullptr;
    if (bits == 8) {
        if (scale_bits == 16) pack_onnx<8, true>(P, qw, scales, nullptr);
        else pack_onnx<8, false>(P, qw, scales, nullptr);
    } else {
        if (scale_bits == 16) pack_onnx<4, true>(P, qw, scales, qzeros);
        else pack_onnx<4, false>(P, qw, scales, qzeros);
    }
    populate_replica(P);
    return P;
}

void i4_free(void* handle) {
    free_packed(static_cast<PackedWeight*>(handle));
}

// C[M,N] = A[M,K] @ dequant(B).T  (row-major). M small; templated so
// accumulators stay in registers. M>8 processed in row-chunks of 8.
void i4_matmul(void* handle, const float* A, int M, float* C) {
    auto* P = static_cast<PackedWeight*>(handle);
    const int K = P->K, nblk = P->nblk;
    const bool hp = g_hi_prec != 0;
    uint8_t* au8 = static_cast<uint8_t*>(aa64((size_t)M * K));
    uint8_t* au8_lo = hp ? static_cast<uint8_t*>(aa64((size_t)M * K)) : nullptr;
    float* ascale = static_cast<float*>(aa64((size_t)M * nblk * sizeof(float)));
    if (hp) quantize_A_i16(A, M, K, nblk, au8, au8_lo, ascale);
    else quantize_A(A, M, K, nblk, au8, ascale);

    auto dispatch = [&](int mm, const uint8_t* a8, const uint8_t* alo, const float* asc, float* c) {
        if (hp) dispatch_run<true>(P, mm, a8, alo, asc, c);
        else dispatch_run<false>(P, mm, a8, alo, asc, c);
    };
    if (M <= 8) {
        dispatch(M, au8, au8_lo, ascale, C);
    } else {
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = (M - m0 < 8) ? (M - m0) : 8;
            const uint8_t* alo = au8_lo ? au8_lo + (size_t)m0 * K : nullptr;
            dispatch(mm, au8 + (size_t)m0 * K, alo, ascale + (size_t)m0 * nblk, C + (size_t)m0 * P->N);
        }
    }
    free(au8); free(au8_lo); free(ascale);
}

// Matmul to be called from INSIDE an active omp parallel region (single
// persistent region for the whole forward -> no per-matmul fork/join, both
// sockets stay hot).  au8/ascale are caller-provided scratch. For hi-prec the
// caller must pass au8 sized 2*M*K (hi in [0,M*K), lo in [M*K,2*M*K)). M<=8.
void i4_matmul_region(void* handle, const float* A, int M, float* C,
                      uint8_t* au8, float* ascale) {
    auto* P = static_cast<PackedWeight*>(handle);
    const bool hp = g_hi_prec != 0;
    uint8_t* au8_lo = au8 + (size_t)M * P->K;
    #pragma omp single
    {
        if (hp) quantize_A_i16(A, M, P->K, P->nblk, au8, au8_lo, ascale);
        else quantize_A(A, M, P->K, P->nblk, au8, ascale);
    }
    // implicit barrier after single: au8/ascale visible to all threads
    if (hp) dispatch_run_region<true>(P, M, au8, au8_lo, ascale, C);
    else dispatch_run_region<false>(P, M, au8, au8_lo, ascale, C);
    #pragma omp barrier
}

}  // extern "C"
