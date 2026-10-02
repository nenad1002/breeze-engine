// Full Phi-3.5 forward pass in C++.  Runs all matmuls back-to-back with the
// VNNI int4 kernel (i4_matmul) and implements embed / rmsnorm / GQA / silu in
// C++ so there is no per-op dispatch overhead and the memory subsystem
// stays streaming. Prefill only (past_len == 0), batch == 1, 1 to 8 tokens.
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

extern "C" void i4_matmul(void* handle, const float* A, int M, float* C);
extern "C" void i4_matmul_region(void* handle, const float* A, int M, float* C,
                                 uint8_t* au8, float* ascale);

static inline float half2float(uint16_t h) { return _cvtsh_ss(h); }

static inline void* aa(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, n)) return nullptr;
    return p;
}

struct Layer {
    void* qkv; void* o; void* gate; void* up; void* down;
    const float* in_ln;   // input_layernorm weight [H]
    const float* post_ln; // post_attention_layernorm weight [H]
};

struct Model {
    int L, H, nh, hs, inter, vocab;
    float eps, scale;
    int rot_dim;            // == hs (96); cos/sin have hs/2 (48) columns
    const uint16_t* embed;  // [vocab, H] fp16
    const float* cosc;      // [max_pos, hs/2]
    const float* sinc;      // [max_pos, hs/2]
    std::vector<Layer> layers;
    const float* final_ln;
    void* lmhead;
};

extern "C" void* i4_model_new(int L, int H, int nh, int hs, int inter,
                              int vocab, float eps, float scale, int rot_dim) {
    Model* m = new Model();
    m->L = L; m->H = H; m->nh = nh; m->hs = hs; m->inter = inter;
    m->vocab = vocab; m->eps = eps; m->scale = scale; m->rot_dim = rot_dim;
    m->layers.resize(L);
    return m;
}

extern "C" void i4_model_set_embed(void* mm, const uint16_t* embed) {
    static_cast<Model*>(mm)->embed = embed;
}
extern "C" void i4_model_set_rotary(void* mm, const float* cosc, const float* sinc) {
    Model* m = static_cast<Model*>(mm);
    m->cosc = cosc; m->sinc = sinc;
}
extern "C" void i4_model_set_layer(void* mm, int l, void* qkv, void* o, void* gate,
                                   void* up, void* down, const float* in_ln,
                                   const float* post_ln) {
    Model* m = static_cast<Model*>(mm);
    m->layers[l] = Layer{qkv, o, gate, up, down, in_ln, post_ln};
}
extern "C" void i4_model_set_final(void* mm, const float* final_ln, void* lmhead) {
    Model* m = static_cast<Model*>(mm);
    m->final_ln = final_ln; m->lmhead = lmhead;
}
extern "C" void i4_model_free(void* mm) { delete static_cast<Model*>(mm); }

// out[s,H] = rmsnorm(x[s,H]) * w
static void rmsnorm(float* out, const float* x, const float* w,
                    int s, int H, float eps) {
    for (int i = 0; i < s; ++i) {
        const float* xr = x + (size_t)i * H;
        double ss = 0.0;
        for (int j = 0; j < H; ++j) ss += (double)xr[j] * xr[j];
        float inv = 1.0f / std::sqrt((float)(ss / H) + eps);
        float* or_ = out + (size_t)i * H;
        for (int j = 0; j < H; ++j) or_[j] = xr[j] * inv * w[j];
    }
}

// resid += add ; out = rmsnorm(resid) * w
static void skip_rmsnorm(float* out, float* resid, const float* add,
                         const float* w, int s, int H, float eps) {
    for (int i = 0; i < s; ++i) {
        float* r = resid + (size_t)i * H;
        const float* a = add + (size_t)i * H;
        double ss = 0.0;
        for (int j = 0; j < H; ++j) { r[j] += a[j]; ss += (double)r[j] * r[j]; }
        float inv = 1.0f / std::sqrt((float)(ss / H) + eps);
        float* o = out + (size_t)i * H;
        for (int j = 0; j < H; ++j) o[j] = r[j] * inv * w[j];
    }
}

// NeoX rotary (interleaved=0) applied in place to a [hs] head vector at pos.
static inline void rotary(float* v, const float* cosr, const float* sinr, int hs) {
    int half = hs / 2;
    for (int d = 0; d < half; ++d) {
        float a = v[d], b = v[d + half];
        float c = cosr[d], s = sinr[d];
        v[d]        = a * c - b * s;
        v[d + half] = b * c + a * s;
    }
}

// GQA (here MHA: nh == kv heads).  qkv[s, 3H] packed [q(H) | k(H) | v(H)].
// out[s, H].  Causal, prefill positions = 0..s-1.
static void gqa(float* out, const float* qkv, const Model* m, int s) {
    const int H = m->H, nh = m->nh, hs = m->hs, half = hs / 2;
    const float scale = m->scale;
    std::vector<float> q(hs), score(s);
    std::vector<float> kbuf((size_t)s * hs), vbuf((size_t)s * hs);
    for (int h = 0; h < nh; ++h) {
        const int qo = h * hs, ko = H + h * hs, vo = 2 * H + h * hs;
        // rotary k, copy v
        for (int j = 0; j < s; ++j) {
            const float* kr = qkv + (size_t)j * 3 * H + ko;
            float* kd = &kbuf[(size_t)j * hs];
            std::memcpy(kd, kr, hs * sizeof(float));
            rotary(kd, m->cosc + (size_t)j * half, m->sinc + (size_t)j * half, hs);
            std::memcpy(&vbuf[(size_t)j * hs], qkv + (size_t)j * 3 * H + vo, hs * sizeof(float));
        }
        for (int i = 0; i < s; ++i) {
            std::memcpy(q.data(), qkv + (size_t)i * 3 * H + qo, hs * sizeof(float));
            rotary(q.data(), m->cosc + (size_t)i * half, m->sinc + (size_t)i * half, hs);
            float mx = -1e30f;
            for (int j = 0; j <= i; ++j) {
                const float* kd = &kbuf[(size_t)j * hs];
                float d = 0.0f;
                for (int t = 0; t < hs; ++t) d += q[t] * kd[t];
                d *= scale; score[j] = d; if (d > mx) mx = d;
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) { score[j] = std::exp(score[j] - mx); sum += score[j]; }
            float invs = 1.0f / sum;
            float* od = out + (size_t)i * H + h * hs;
            for (int t = 0; t < hs; ++t) od[t] = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float w = score[j] * invs;
                const float* vd = &vbuf[(size_t)j * hs];
                for (int t = 0; t < hs; ++t) od[t] += w * vd[t];
            }
        }
    }
}

// gate = silu(gate) * up
static void silu_mul(float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        gate[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

// logits[s, vocab]
extern "C" void i4_forward(void* mm, const int64_t* ids, int s, float* logits) {
    Model* m = static_cast<Model*>(mm);
    const int H = m->H, inter = m->inter;
    float* resid = (float*)aa((size_t)s * H * sizeof(float));
    float* h     = (float*)aa((size_t)s * H * sizeof(float));
    float* qkv   = (float*)aa((size_t)s * 3 * H * sizeof(float));
    float* attn  = (float*)aa((size_t)s * H * sizeof(float));
    float* tmp   = (float*)aa((size_t)s * H * sizeof(float));
    float* gate  = (float*)aa((size_t)s * inter * sizeof(float));
    float* up    = (float*)aa((size_t)s * inter * sizeof(float));
    float* mlp   = (float*)aa((size_t)s * H * sizeof(float));
    // matmul A-quant scratch (max K == inter for down_proj)
    int maxK = inter > 3 * H ? inter : 3 * H;
    int maxNblk = maxK / 32;
    uint8_t* au8 = (uint8_t*)aa(2 * (size_t)s * maxK);
    float* ascale = (float*)aa((size_t)s * maxNblk * sizeof(float));

    // Single persistent parallel region for the WHOLE forward: no per-matmul
    // fork/join, so both sockets stay hot across all 161 matmuls.  Serial ops
    // (embed / rmsnorm / gqa / silu) run in `omp single` (cheap for M=few).
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < s; ++i) {
                const uint16_t* row = m->embed + (size_t)ids[i] * H;
                for (int j = 0; j < H; ++j) resid[(size_t)i * H + j] = half2float(row[j]);
            }
            rmsnorm(h, resid, m->layers[0].in_ln, s, H, m->eps);
        }
        for (int l = 0; l < m->L; ++l) {
            const Layer& L = m->layers[l];
            if (l > 0) {
                #pragma omp single
                skip_rmsnorm(h, resid, mlp, L.in_ln, s, H, m->eps);
            }
            i4_matmul_region(L.qkv, h, s, qkv, au8, ascale);
            #pragma omp single
            gqa(attn, qkv, m, s);
            i4_matmul_region(L.o, attn, s, tmp, au8, ascale);
            #pragma omp single
            skip_rmsnorm(h, resid, tmp, L.post_ln, s, H, m->eps);
            i4_matmul_region(L.gate, h, s, gate, au8, ascale);
            i4_matmul_region(L.up, h, s, up, au8, ascale);
            #pragma omp single
            silu_mul(gate, up, (size_t)s * inter);
            i4_matmul_region(L.down, gate, s, mlp, au8, ascale);
        }
        #pragma omp single
        skip_rmsnorm(h, resid, mlp, m->final_ln, s, H, m->eps);
        i4_matmul_region(m->lmhead, h, s, logits, au8, ascale);
    }

    free(resid); free(h); free(qkv); free(attn); free(tmp);
    free(gate); free(up); free(mlp); free(au8); free(ascale);
}

