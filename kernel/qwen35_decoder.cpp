// Qwen3.5 GatedDeltaNet ops in C++ so the hot LinearAttention runs on the
// OpenMP threads with no per-op dispatch overhead.
// gated_delta only; prefill/decode both (state carried via past/present).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include <omp.h>

// Depthwise causal 1D conv with carry state (ndim=1). x:[B,C,L] w:[C,1,K]
// bias:[C]|null past:[B,C,K-1]|null -> output:[B,C,L] present:[B,C,K-1].
extern "C" void i4_causal_conv(
    const float* x, const float* w, const float* bias, const float* past,
    float* output, float* present, int B, int C, int L, int K, int silu) {
    const int pad = K - 1;
    #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < B * C; ++bc) {
        const int c = bc % C;
        const float* xr = x + (size_t)bc * L;
        const float* wr = w + (size_t)c * K;
        const float* pr = past ? past + (size_t)bc * pad : nullptr;
        float* orow = output + (size_t)bc * L;
        float* prow = present + (size_t)bc * pad;
        float bval = bias ? bias[c] : 0.0f;
        // Read the conceptual [past, x] window directly. No fixed prefill
        // buffer and no per-channel allocation in the one-token decode path.
        auto sample = [&](int index) {
            return index < pad ? (pr ? pr[index] : 0.0f) : xr[index - pad];
        };
        for (int l = 0; l < L; ++l) {
            float s = bval;
            for (int kk = 0; kk < K; ++kk) s += wr[kk] * sample(l + kk);
            if (silu) s = s / (1.0f + std::exp(-s));
            orow[l] = s;
        }
        for (int i = 0; i < pad; ++i) prow[i] = sample(L + i);
    }
}

// q:[B,T,Hq*dk] k:[B,T,n_k*dk] v:[B,T,Hkv*dv] past/present:[B,Hkv,dk,dv]
// decay:[B,T,Hkv*dk] (per-key-dim) or [B,T,Hkv]; beta:[B,T,Hkv] or [B,T,1]
// output:[B,T,max(Hq,Hkv)*dv].
extern "C" void i4_linear_attention(
    const float* q, const float* k, const float* v, const float* past,
    const float* decay, const float* beta, float* output, float* present,
    int B, int T, int Hq, int Hkv, int dk, int dv, int n_k,
    int decay_per_key_dim, int beta_per_head, float scale) {
    const int kv_per_k = Hkv / n_k;
    const int hpg = (Hq >= Hkv) ? (Hq / Hkv) : 0;   // 0 => inverse GQA
    const int out_hidden = (Hq > Hkv ? Hq : Hkv) * dv;
    const size_t sper = (size_t)dk * dv;

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < B * Hkv; ++task) {
        const int b = task / Hkv;
        const int h_kv = task % Hkv;
        const int h_k = h_kv / kv_per_k;
        float* S = present + (size_t)task * sper;
        if (past) std::memcpy(S, past + (size_t)task * sper, sper * sizeof(float));
        else std::memset(S, 0, sper * sizeof(float));

        float retrieved[256];  // dv <= 256
        for (int t = 0; t < T; ++t) {
            const float* kt = k + ((size_t)(b * T + t) * n_k + h_k) * dk;
            const float* vt = v + ((size_t)(b * T + t) * Hkv + h_kv) * dv;
            // decay
            if (decay_per_key_dim) {
                const float* g = decay + ((size_t)(b * T + t) * Hkv + h_kv) * dk;
                for (int i = 0; i < dk; ++i) {
                    float e = std::exp(g[i]);
                    float* sr = S + (size_t)i * dv;
                    for (int j = 0; j < dv; ++j) sr[j] *= e;
                }
            } else {
                float e = std::exp(decay[(size_t)(b * T + t) * Hkv + h_kv]);
                for (size_t i = 0; i < sper; ++i) S[i] *= e;
            }
            // retrieved = S^T k
            for (int j = 0; j < dv; ++j) retrieved[j] = 0.0f;
            for (int i = 0; i < dk; ++i) {
                float ki = kt[i]; const float* sr = S + (size_t)i * dv;
                for (int j = 0; j < dv; ++j) retrieved[j] += sr[j] * ki;
            }
            // delta = beta*(v - retrieved);  S += k (x) delta
            float bt = beta_per_head ? beta[(size_t)(b * T + t) * Hkv + h_kv]
                                     : beta[(size_t)(b * T + t)];
            for (int j = 0; j < dv; ++j) retrieved[j] = bt * (vt[j] - retrieved[j]);
            for (int i = 0; i < dk; ++i) {
                float ki = kt[i]; float* sr = S + (size_t)i * dv;
                for (int j = 0; j < dv; ++j) sr[j] += ki * retrieved[j];
            }
            // readout
            if (hpg > 0) {
                for (int g = 0; g < hpg; ++g) {
                    int h_q = h_kv * hpg + g;
                    const float* qt = q + ((size_t)(b * T + t) * Hq + h_q) * dk;
                    float* ot = output + ((size_t)(b * T + t) * out_hidden) + (size_t)h_q * dv;
                    for (int j = 0; j < dv; ++j) ot[j] = 0.0f;
                    for (int i = 0; i < dk; ++i) {
                        float qi = qt[i]; const float* sr = S + (size_t)i * dv;
                        for (int j = 0; j < dv; ++j) ot[j] += qi * sr[j];
                    }
                    for (int j = 0; j < dv; ++j) ot[j] *= scale;
                }
            } else {
                int h_q = h_kv * Hq / Hkv;
                const float* qt = q + ((size_t)(b * T + t) * Hq + h_q) * dk;
                float* ot = output + ((size_t)(b * T + t) * out_hidden) + (size_t)h_kv * dv;
                for (int j = 0; j < dv; ++j) ot[j] = 0.0f;
                for (int i = 0; i < dk; ++i) {
                    float qi = qt[i]; const float* sr = S + (size_t)i * dv;
                    for (int j = 0; j < dv; ++j) ot[j] += qi * sr[j];
                }
                for (int j = 0; j < dv; ++j) ot[j] *= scale;
            }
        }
    }
}

// ============================================================ full Qwen3.5 forward
extern "C" void i4_matmul(void* handle, const float* A, int M, float* C);

namespace {
inline float half2f(uint16_t h) { return _cvtsh_ss(h); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }
inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
inline void* aq(size_t n) { void* p = nullptr; return posix_memalign(&p, 64, n) ? nullptr : p; }

// out[s,H] = x/sqrt(mean(x^2)+eps) * w
void rmsnorm(float* out, const float* x, const float* w, int s, int H, float eps) {
    for (int i = 0; i < s; ++i) {
        const float* xr = x + (size_t)i * H; float* o = out + (size_t)i * H;
        double ss = 0; for (int j = 0; j < H; ++j) ss += (double)xr[j] * xr[j];
        float inv = 1.0f / std::sqrt((float)(ss / H) + eps);
        for (int j = 0; j < H; ++j) o[j] = xr[j] * inv * w[j];
    }
}

struct QLin {
    void *in_qkv, *in_z, *in_b, *in_a, *out_proj, *mg, *mu, *md;
    const float *input_ln, *conv_w, *conv_b, *neg_exp_A, *dt_bias, *gnorm_w, *post_ln;
};
struct QFull {
    void *q, *k, *v, *o, *mg, *mu, *md;
    const float *input_ln, *q_norm, *k_norm, *post_ln;
};
struct QModel {
    int L, H, vocab, inter;
    int Hk, Hv, head_k, head_v, key_dim, value_dim, conv_dim, conv_k;
    int n_heads, n_kv, head_dim, rotary_dim; float attn_scale, eps;
    const uint16_t* embed;
    const float *cosc, *sinc;             // [max_pos, rotary_dim/2]
    std::vector<int> is_full;
    std::vector<QLin> lin; std::vector<QFull> full;
    const float* final_ln; void* lmhead;
    // decode state (one active sequence): linear=conv+recurrent, full=KV cache
    int max_seq; bool st_alloc;
    std::vector<float*> conv_st, recur_st, kcache, vcache;
};
}  // namespace

extern "C" void* i4_qwen_new(int L, int H, int vocab, int inter, int Hk, int Hv,
                             int head_k, int head_v, int conv_k, int n_heads, int n_kv,
                             int head_dim, int rotary_dim, float attn_scale, float eps, int max_seq) {
    QModel* m = new QModel();
    m->L = L; m->H = H; m->vocab = vocab; m->inter = inter;
    m->Hk = Hk; m->Hv = Hv; m->head_k = head_k; m->head_v = head_v;
    m->key_dim = Hk * head_k; m->value_dim = Hv * head_v;
    m->conv_dim = m->key_dim * 2 + m->value_dim; m->conv_k = conv_k;
    m->n_heads = n_heads; m->n_kv = n_kv; m->head_dim = head_dim;
    m->rotary_dim = rotary_dim; m->attn_scale = attn_scale; m->eps = eps;
    m->max_seq = max_seq; m->st_alloc = false;
    m->is_full.assign(L, 0); m->lin.resize(L); m->full.resize(L);
    return m;
}
extern "C" void i4_qwen_set_embed(void* mm, const uint16_t* e) { ((QModel*)mm)->embed = e; }
extern "C" void i4_qwen_set_rotary(void* mm, const float* c, const float* s) {
    QModel* m = (QModel*)mm; m->cosc = c; m->sinc = s;
}
extern "C" void i4_qwen_set_linear(void* mm, int l, void* in_qkv, void* in_z, void* in_b,
        void* in_a, void* out_proj, void* mg, void* mu, void* md,
        const float* input_ln, const float* conv_w, const float* conv_b,
        const float* neg_exp_A, const float* dt_bias, const float* gnorm_w, const float* post_ln) {
    QModel* m = (QModel*)mm; m->is_full[l] = 0;
    m->lin[l] = QLin{in_qkv, in_z, in_b, in_a, out_proj, mg, mu, md,
                     input_ln, conv_w, conv_b, neg_exp_A, dt_bias, gnorm_w, post_ln};
}
extern "C" void i4_qwen_set_full(void* mm, int l, void* q, void* k, void* v, void* o,
        void* mg, void* mu, void* md, const float* input_ln, const float* q_norm,
        const float* k_norm, const float* post_ln) {
    QModel* m = (QModel*)mm; m->is_full[l] = 1;
    m->full[l] = QFull{q, k, v, o, mg, mu, md, input_ln, q_norm, k_norm, post_ln};
}
extern "C" void i4_qwen_set_final(void* mm, const float* final_ln, void* lmhead) {
    QModel* m = (QModel*)mm; m->final_ln = final_ln; m->lmhead = lmhead;
}
extern "C" void i4_qwen_free(void* mm) {
    QModel* m = (QModel*)mm;
    if (m->st_alloc) for (int l = 0; l < m->L; ++l) {
        free(m->conv_st[l]); free(m->recur_st[l]); free(m->kcache[l]); free(m->vcache[l]);
    }
    delete m;
}

// forward declaration of the gated-delta / conv kernels (defined above)
extern "C" void i4_linear_attention(const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, int, int, int, int, int, int, int, int, int, float);
extern "C" void i4_causal_conv(const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int, int);

// Version 2 adds safe arbitrary-length conv and a checked forward return code.
extern "C" int i4_qwen_abi_version() { return 2; }

extern "C" int i4_qwen_forward(void* mm, const float* inputs_embeds, int s, int past_len, float* logits) {
    QModel* m = (QModel*)mm;
    if (!m || !inputs_embeds || !logits || s <= 0 || past_len < 0 ||
        s > m->max_seq || past_len > m->max_seq - s) return -1;
    const int H = m->H, inter = m->inter;
    const float eps = m->eps;
    auto A = [&](size_t n) { return (float*)aq(n * sizeof(float)); };
    float* h = A((size_t)s * H);            // residual stream
    float* hn = A((size_t)s * H);           // normed
    float* mix = A((size_t)s * m->conv_dim);
    float* z = A((size_t)s * m->value_dim);
    float* bproj = A((size_t)s * m->Hv);
    float* aproj = A((size_t)s * m->Hv);
    float* convin = A((size_t)s * m->conv_dim);
    float* convout = A((size_t)s * m->conv_dim);
    float* convpresent = A((size_t)m->conv_dim * (m->conv_k - 1));
    float* qbuf = A((size_t)s * m->key_dim);
    float* kbuf = A((size_t)s * m->key_dim);
    // Attention inner width is not hidden_size (27B: 6144 vs 5120).
    float* attn = A((size_t)s * std::max(m->value_dim, m->n_heads * m->head_dim));
    float* gbuf = A((size_t)s * m->Hv);
    float* betab = A((size_t)s * m->Hv);
    float* lstate = A((size_t)m->Hv * m->head_k * m->head_v);
    float* mixer = A((size_t)s * H);
    float* qkv = A((size_t)s * (size_t)m->n_heads * m->head_dim * 2);
    float* kf = A((size_t)s * (size_t)m->n_kv * m->head_dim);
    float* vf = A((size_t)s * (size_t)m->n_kv * m->head_dim);
    float* gate = A((size_t)s * H);
    float* g1 = A((size_t)s * inter);
    float* g2 = A((size_t)s * inter);

    // inputs_embeds provided directly (this decoder has no embedding table)
    std::memcpy(h, inputs_embeds, (size_t)s * H * sizeof(float));

    // lazy-allocate per-layer decode state
    if (!m->st_alloc) {
        m->conv_st.assign(m->L, nullptr); m->recur_st.assign(m->L, nullptr);
        m->kcache.assign(m->L, nullptr); m->vcache.assign(m->L, nullptr);
        for (int l = 0; l < m->L; ++l) {
            if (m->is_full[l]) {
                m->kcache[l] = (float*)aq((size_t)m->n_kv * m->max_seq * m->head_dim * sizeof(float));
                m->vcache[l] = (float*)aq((size_t)m->n_kv * m->max_seq * m->head_dim * sizeof(float));
            } else {
                m->conv_st[l] = (float*)aq((size_t)m->conv_dim * (m->conv_k - 1) * sizeof(float));
                m->recur_st[l] = (float*)aq((size_t)m->Hv * m->head_k * m->head_v * sizeof(float));
            }
        }
        m->st_alloc = true;
    }
    if (past_len == 0) {   // new sequence: clear recurrent/conv states
        for (int l = 0; l < m->L; ++l) if (!m->is_full[l]) {
            std::memset(m->conv_st[l], 0, (size_t)m->conv_dim * (m->conv_k - 1) * sizeof(float));
            std::memset(m->recur_st[l], 0, (size_t)m->Hv * m->head_k * m->head_v * sizeof(float));
        }
    }

    static int g_maxl = -2;
    if (g_maxl == -2) { const char* e = getenv("I4_QWEN_MAXL"); g_maxl = e ? atoi(e) : -1; }
    const int Lrun = (g_maxl >= 0) ? std::min(g_maxl, m->L) : m->L;

    for (int l = 0; l < Lrun; ++l) {
        rmsnorm(hn, h, m->is_full[l] ? m->full[l].input_ln : m->lin[l].input_ln, s, H, eps);

        if (!m->is_full[l]) {
            const QLin& L = m->lin[l];
            const int Hk = m->Hk, Hv = m->Hv, hk = m->head_k, hv = m->head_v;
            const int kd = m->key_dim, vd = m->value_dim, cd = m->conv_dim;
            i4_matmul(L.in_qkv, hn, s, mix);
            i4_matmul(L.in_z, hn, s, z);
            i4_matmul(L.in_b, hn, s, bproj);
            i4_matmul(L.in_a, hn, s, aproj);
            // conv over channels: [s,cd] -> [cd,s]
            for (int i = 0; i < s; ++i) for (int c = 0; c < cd; ++c) convin[(size_t)c * s + i] = mix[(size_t)i * cd + c];
            i4_causal_conv(convin, L.conv_w, L.conv_b, m->conv_st[l], convout, convpresent, 1, cd, s, m->conv_k, 1);
            std::memcpy(m->conv_st[l], convpresent, (size_t)cd * (m->conv_k - 1) * sizeof(float));
            for (int i = 0; i < s; ++i) for (int c = 0; c < cd; ++c) mix[(size_t)i * cd + c] = convout[(size_t)c * s + i];
            // split q,k,v and L2-norm q,k per head
            for (int i = 0; i < s; ++i) {
                const float* row = mix + (size_t)i * cd;
                for (int hh = 0; hh < Hk; ++hh) {
                    const float* qp = row + hh * hk; float* qo = qbuf + (size_t)i * kd + hh * hk;
                    const float* kp = row + kd + hh * hk; float* ko = kbuf + (size_t)i * kd + hh * hk;
                    double sq = 0, sk = 0;
                    for (int d = 0; d < hk; ++d) { sq += (double)qp[d] * qp[d]; sk += (double)kp[d] * kp[d]; }
                    float iq = 1.0f / std::sqrt((float)sq + eps), ik = 1.0f / std::sqrt((float)sk + eps);
                    for (int d = 0; d < hk; ++d) { qo[d] = qp[d] * iq; ko[d] = kp[d] * ik; }
                }
                // beta, g
                for (int hh = 0; hh < Hv; ++hh) {
                    betab[(size_t)i * Hv + hh] = sigmoidf(bproj[(size_t)i * Hv + hh]);
                    gbuf[(size_t)i * Hv + hh] = L.neg_exp_A[hh] * softplusf(aproj[(size_t)i * Hv + hh] + L.dt_bias[hh]);
                }
            }
            // value = mix[:, 2*kd : 2*kd+vd]; gather into convin (free now) to avoid aliasing attn
            float* vptr = convin;
            for (int i = 0; i < s; ++i)
                std::memcpy(vptr + (size_t)i * vd, mix + (size_t)i * cd + 2 * kd, vd * sizeof(float));
            i4_linear_attention(qbuf, kbuf, vptr, m->recur_st[l], gbuf, betab, attn, lstate,
                                1, s, Hk, Hv, hk, hv, Hk, /*decay_per_key*/0, /*beta_per_head*/1,
                                1.0f / std::sqrt((float)hk));
            std::memcpy(m->recur_st[l], lstate, (size_t)Hv * hk * hv * sizeof(float));
            // gated norm per head_v: rmsnorm(attn_h)*gnorm_w*silu(z_h)  -> reuse hn as gn buffer[s,vd]
            for (int i = 0; i < s; ++i) {
                for (int hh = 0; hh < Hv; ++hh) {
                    float* ar = attn + (size_t)i * vd + hh * hv;
                    const float* zr = z + (size_t)i * vd + hh * hv;
                    double ss = 0; for (int d = 0; d < hv; ++d) ss += (double)ar[d] * ar[d];
                    float inv = 1.0f / std::sqrt((float)(ss / hv) + eps);
                    for (int d = 0; d < hv; ++d) ar[d] = ar[d] * inv * L.gnorm_w[d] * siluf(zr[d]);
                }
            }
            i4_matmul(L.out_proj, attn, s, mixer);
        } else {
            const QFull& F = m->full[l];
            const int nh = m->n_heads, nkv = m->n_kv, hd = m->head_dim, rd = m->rotary_dim, half = rd / 2;
            float* kc = m->kcache[l]; float* vc = m->vcache[l];   // [nkv, max_seq, hd]
            i4_matmul(F.q, hn, s, qkv);          // [s, nh*hd*2]
            i4_matmul(F.k, hn, s, kf);           // [s, nkv*hd]
            i4_matmul(F.v, hn, s, vf);           // [s, nkv*hd]
            // q_norm+rope(q), k_norm+rope(k), append k,v to cache at absolute pos
            for (int i = 0; i < s; ++i) {
                const int pos = past_len + i;
                const float* cosr = m->cosc + (size_t)pos * half;
                const float* sinr = m->sinc + (size_t)pos * half;
                for (int hh = 0; hh < nh; ++hh) {
                    float* q = qkv + (size_t)i * nh * hd * 2 + (size_t)hh * hd * 2;
                    double ss = 0; for (int d = 0; d < hd; ++d) ss += (double)q[d] * q[d];
                    float inv = 1.0f / std::sqrt((float)(ss / hd) + eps);
                    for (int d = 0; d < hd; ++d) q[d] = q[d] * inv * F.q_norm[d];
                    for (int d = 0; d < half; ++d) {
                        float a0 = q[d], a1 = q[d + half];
                        q[d] = a0 * cosr[d] - a1 * sinr[d]; q[d + half] = a1 * cosr[d] + a0 * sinr[d];
                    }
                }
                for (int hh = 0; hh < nkv; ++hh) {
                    float* k = kf + (size_t)i * nkv * hd + (size_t)hh * hd;
                    double ss = 0; for (int d = 0; d < hd; ++d) ss += (double)k[d] * k[d];
                    float inv = 1.0f / std::sqrt((float)(ss / hd) + eps);
                    for (int d = 0; d < hd; ++d) k[d] = k[d] * inv * F.k_norm[d];
                    for (int d = 0; d < half; ++d) {
                        float a0 = k[d], a1 = k[d + half];
                        k[d] = a0 * cosr[d] - a1 * sinr[d]; k[d + half] = a1 * cosr[d] + a0 * sinr[d];
                    }
                    std::memcpy(kc + ((size_t)hh * m->max_seq + pos) * hd, k, hd * sizeof(float));
                    std::memcpy(vc + ((size_t)hh * m->max_seq + pos) * hd,
                                vf + (size_t)i * nkv * hd + (size_t)hh * hd, hd * sizeof(float));
                }
            }
            // GQA SDPA over the cache (causal), then output gate — parallel over heads
            const int grp = nh / nkv;
            const int sclen = past_len + s;
            float* sc_all = A((size_t)nh * sclen);   // per-head scratch, no malloc in loop
            #pragma omp parallel for schedule(static)
            for (int hh = 0; hh < nh; ++hh) {
                int kvh = hh / grp;
                float* sc = sc_all + (size_t)hh * sclen;
                for (int i = 0; i < s; ++i) {
                    const int qpos = past_len + i;
                    const float* q = qkv + (size_t)i * nh * hd * 2 + (size_t)hh * hd * 2;
                    float mx = -1e30f;
                    for (int j = 0; j <= qpos; ++j) {
                        const float* k = kc + ((size_t)kvh * m->max_seq + j) * hd;
                        float d = 0; for (int t = 0; t < hd; ++t) d += q[t] * k[t];
                        d *= m->attn_scale; sc[j] = d; if (d > mx) mx = d;
                    }
                    float sm = 0; for (int j = 0; j <= qpos; ++j) { sc[j] = std::exp(sc[j] - mx); sm += sc[j]; }
                    float ism = 1.0f / sm;
                    float* out = attn + (size_t)i * (nh * hd) + (size_t)hh * hd;
                    for (int t = 0; t < hd; ++t) out[t] = 0;
                    for (int j = 0; j <= qpos; ++j) {
                        float w = sc[j] * ism; const float* vv = vc + ((size_t)kvh * m->max_seq + j) * hd;
                        for (int t = 0; t < hd; ++t) out[t] += w * vv[t];
                    }
                    const float* gt = q + hd;
                    for (int t = 0; t < hd; ++t) out[t] *= sigmoidf(gt[t]);
                }
            }
            i4_matmul(F.o, attn, s, mixer);
            free(sc_all);
        }
        // residual + mixer
        for (size_t i = 0; i < (size_t)s * H; ++i) h[i] += mixer[i];
        // MLP
        void* mg = m->is_full[l] ? m->full[l].mg : m->lin[l].mg;
        void* mu = m->is_full[l] ? m->full[l].mu : m->lin[l].mu;
        void* md = m->is_full[l] ? m->full[l].md : m->lin[l].md;
        const float* post_ln = m->is_full[l] ? m->full[l].post_ln : m->lin[l].post_ln;
        rmsnorm(hn, h, post_ln, s, H, eps);
        i4_matmul(mg, hn, s, g1);
        i4_matmul(mu, hn, s, g2);
        for (size_t i = 0; i < (size_t)s * inter; ++i) g1[i] = siluf(g1[i]) * g2[i];
        i4_matmul(md, g1, s, mixer);
        for (size_t i = 0; i < (size_t)s * H; ++i) h[i] += mixer[i];
    }
    if (g_maxl >= 0) {   // debug: emit hidden state after Lrun layers
        std::memcpy(logits, h, (size_t)s * H * sizeof(float));
        free(h); free(hn); free(mix); free(z); free(bproj); free(aproj); free(convin);
        free(convout); free(convpresent); free(qbuf); free(kbuf); free(attn); free(gbuf);
        free(betab); free(lstate); free(mixer); free(qkv); free(kf); free(vf); free(gate);
        free(g1); free(g2);
        return 0;
    }
    rmsnorm(hn, h, m->final_ln, s, H, eps);
    i4_matmul(m->lmhead, hn, s, logits);

    free(h); free(hn); free(mix); free(z); free(bproj); free(aproj); free(convin);
    free(convout); free(convpresent); free(qbuf); free(kbuf); free(attn); free(gbuf);
    free(betab); free(lstate); free(mixer); free(qkv); free(kf); free(vf); free(gate);
    free(g1); free(g2);
    return 0;
}

