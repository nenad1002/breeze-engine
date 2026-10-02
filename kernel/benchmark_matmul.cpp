// Standalone driver to profile the int4 kernel with perf.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>

extern "C" {
void* i4_prepack(const uint8_t*, const float*, const int8_t*, const int32_t*, int, int, int, int);
void i4_matmul(void*, const float*, int, float*);
void i4_set_threads(int);
void i4_free(void*);
}

int main(int argc, char** argv) {
    int N = 9216, K = 3072, nblk = K / 32, ntiles = N / 16, M = 5;
    int th = argc > 1 ? atoi(argv[1]) : 24;
    int reps = argc > 2 ? atoi(argv[2]) : 300;
    i4_set_threads(th);

    size_t Bsz = (size_t)ntiles * nblk * 8 * 32, Ssz = (size_t)ntiles * nblk * 16;
    uint8_t* B = (uint8_t*)malloc(Bsz);
    for (size_t i = 0; i < Bsz; i++) B[i] = (uint8_t)(i * 131 + 7);
    float* S = (float*)malloc(Ssz * 4);
    for (size_t i = 0; i < Ssz; i++) S[i] = 0.01f;
    int8_t* Z = (int8_t*)malloc(Ssz);
    for (size_t i = 0; i < Ssz; i++) Z[i] = 8;
    int32_t* Corr = (int32_t*)calloc(Ssz, 4);
    void* h = i4_prepack(B, S, Z, Corr, N, K, nblk, 4);
    if (!h) {
        fprintf(stderr, "Could not prepack benchmark weights\n");
        free(B); free(S); free(Z); free(Corr);
        return 1;
    }

    float* A = (float*)malloc((size_t)M * K * 4);
    for (int i = 0; i < M * K; i++) A[i] = 0.1f;
    float* C = (float*)malloc((size_t)M * N * 4);

    for (int i = 0; i < 30; i++) i4_matmul(h, A, M, C);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) i4_matmul(h, A, M, C);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    printf("threads=%d M=%d: %.3f ms  %.1f GB/s\n", th, M, ms, (N * K / 2) / (ms / 1000) / 1e9);
    i4_free(h);
    free(B); free(S); free(Z); free(Corr); free(A); free(C);
    return 0;
}
