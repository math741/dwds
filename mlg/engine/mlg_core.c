/* The Resident Core on the headset: the byte-level mixture-of-experts
 * Transformer of mlgsub/brain.py, run one context at a time with a KV cache.
 * Prompts go through as one batch (each weight row read once for all of
 * them); generation then costs one pass over the active weights per byte.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mlg.h"

typedef struct {
    int rows, cols, q8;
    const float *f;      /* f32 rows */
    const float *scale;  /* q8: one scale per row */
    const int8_t *q;     /* q8 rows */
} mat;

typedef struct {
    const float *ln1w, *ln1b, *ln2w, *ln2b, *gate;
    mat qkv, proj, *up, *down;
} layer;

struct mlg_core {
    mlg_core_info info;
    unsigned char *blob;
    const float *tok, *pos, *lnw, *lnb;
    layer *layers;
    int n;               /* bytes in the context */
    uint8_t *hist;       /* those bytes, to slide the context when it fills */
    float *k, *v;        /* cache [layer][ctx][d] */
    /* scratch for a batch of up to ctx bytes */
    float *x, *h, *qkvb, *att, *tmp, *ff, *ffo, *probs;
    int *idx;
};

/* ---- weights ------------------------------------------------------------- */
typedef struct {
    unsigned char *p, *end;
} reader;

static const void *take(reader *r, size_t bytes) {
    if ((size_t)(r->end - r->p) < bytes) return NULL;
    const void *out = r->p;
    r->p += bytes;
    return out;
}

static int take_mat(reader *r, mat *m, int rows, int cols, int q8) {
    m->rows = rows, m->cols = cols, m->q8 = q8;
    if (q8) {
        m->scale = take(r, sizeof(float) * rows);
        m->q = take(r, (size_t)rows * cols);
        return m->scale && m->q;
    }
    m->f = take(r, sizeof(float) * rows * cols);
    return m->f != NULL;
}

/* y[t][r] = W[r] . x[t] for n inputs of W->cols, outputs of W->rows */
static void matmul(const mat *W, const float *x, float *y, int n) {
    const int R = W->rows, C = W->cols;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((long)R * C * n > (1L << 15))
#endif
    for (int r = 0; r < R; r++) {
        if (W->q8) {
            const int8_t *q = W->q + (size_t)r * C;
            const float s = W->scale[r];
            for (int t = 0; t < n; t++) {
                const float *xt = x + (size_t)t * C;
                float acc = 0.f;
                for (int i = 0; i < C; i++) acc += (float)q[i] * xt[i];
                y[(size_t)t * R + r] = acc * s;
            }
        } else {
            const float *w = W->f + (size_t)r * C;
            for (int t = 0; t < n; t++) {
                const float *xt = x + (size_t)t * C;
                float acc = 0.f;
                for (int i = 0; i < C; i++) acc += w[i] * xt[i];
                y[(size_t)t * R + r] = acc;
            }
        }
    }
}

static void layernorm(const float *x, const float *w, const float *b, float *out, int d) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
    const float inv = 1.f / sqrtf(var / d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

mlg_core *mlg_core_open(const char *path, char *err, int errlen) {
#define FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); mlg_core_close(c); return NULL; } while (0)
    mlg_core *c = calloc(1, sizeof *c);
    FILE *fh = fopen(path, "rb");
    if (!c || !fh) {
        if (fh) fclose(fh);
        FAIL("cannot open %s", path);
    }
    fseek(fh, 0, SEEK_END);
    long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    c->blob = malloc(size > 0 ? size : 1);
    if (!c->blob || fread(c->blob, 1, size, fh) != (size_t)size) {
        fclose(fh);
        FAIL("cannot read %s", path);
    }
    fclose(fh);
    reader r = {c->blob, c->blob + size};
    const uint32_t *hdr = take(&r, 40);
    if (!hdr || memcmp(hdr, "MLGB", 4) || hdr[1] != 1) FAIL("%s is not an MLGB v1 brain", path);
    mlg_core_info *I = &c->info;
    I->d_model = hdr[2], I->n_layers = hdr[3], I->n_heads = hdr[4], I->n_experts = hdr[5];
    I->top_k = hdr[6], I->ffn_mult = hdr[7], I->ctx = hdr[8], I->wtype = hdr[9];
    const int d = I->d_model, L = I->n_layers, E = I->n_experts, F = I->ffn_mult * d, T = I->ctx;
    if (d <= 0 || L <= 0 || E <= 0 || T <= 0 || I->n_heads <= 0 || d % I->n_heads ||
        I->top_k < 1 || I->top_k > E || I->wtype > 1)
        FAIL("bad brain header");
    const int q8 = I->wtype == 1;
    c->tok = take(&r, sizeof(float) * 256 * d);
    c->pos = take(&r, sizeof(float) * T * d);
    c->layers = calloc(L, sizeof(layer));
    if (!c->tok || !c->pos || !c->layers) FAIL("truncated brain");
    long long params = 256LL * d + (long long)T * d + 2 * d;
    for (int l = 0; l < L; l++) {
        layer *y = &c->layers[l];
        y->ln1w = take(&r, 4 * d), y->ln1b = take(&r, 4 * d);
        if (!take_mat(&r, &y->qkv, 3 * d, d, q8) || !take_mat(&r, &y->proj, d, d, q8)) FAIL("truncated brain");
        y->ln2w = take(&r, 4 * d), y->ln2b = take(&r, 4 * d);
        if (E > 1 && !(y->gate = take(&r, sizeof(float) * E * d))) FAIL("truncated brain");
        y->up = calloc(E, sizeof(mat)), y->down = calloc(E, sizeof(mat));
        if (!y->ln1w || !y->ln1b || !y->ln2w || !y->ln2b || !y->up || !y->down) FAIL("truncated brain");
        for (int e = 0; e < E; e++)
            if (!take_mat(&r, &y->up[e], F, d, q8) || !take_mat(&r, &y->down[e], d, F, q8)) FAIL("truncated brain");
        params += 4LL * d * d + 4 * d + (E > 1 ? (long long)E * d : 0) + (long long)E * 2 * F * d;
    }
    c->lnw = take(&r, 4 * d), c->lnb = take(&r, 4 * d);
    if (!c->lnw || !c->lnb || r.p != r.end) FAIL("brain size does not match its header");
    I->params = params;
    c->hist = malloc(T);
    c->k = malloc(sizeof(float) * L * T * d);
    c->v = malloc(sizeof(float) * L * T * d);
    c->x = malloc(sizeof(float) * T * d);
    c->h = malloc(sizeof(float) * T * d);
    c->qkvb = malloc(sizeof(float) * T * 3 * d);
    c->att = malloc(sizeof(float) * T * d);
    c->tmp = malloc(sizeof(float) * T * (d > 256 ? d : 256));
    c->ff = malloc(sizeof(float) * T * F);
    c->ffo = malloc(sizeof(float) * T * d);
    c->probs = malloc(sizeof(float) * T * E);
    c->idx = malloc(sizeof(int) * T * (E + 1));
    if (!c->hist || !c->k || !c->v || !c->x || !c->h || !c->qkvb || !c->att || !c->tmp || !c->ff ||
        !c->ffo || !c->probs || !c->idx)
        FAIL("out of memory for a context of %d bytes", T);
    return c;
#undef FAIL
}

void mlg_core_close(mlg_core *c) {
    if (!c) return;
    if (c->layers)
        for (int l = 0; l < c->info.n_layers; l++) free(c->layers[l].up), free(c->layers[l].down);
    free(c->layers), free(c->blob), free(c->hist), free(c->k), free(c->v), free(c->x), free(c->h);
    free(c->qkvb), free(c->att), free(c->tmp), free(c->ff), free(c->ffo), free(c->probs), free(c->idx);
    free(c);
}

mlg_core_info mlg_core_get_info(const mlg_core *c) { return c->info; }
void mlg_core_reset(mlg_core *c) { c->n = 0; }

/* ---- forward ------------------------------------------------------------- */
static void moe(mlg_core *c, const layer *y, int n) {
    const int d = c->info.d_model, E = c->info.n_experts, K = c->info.top_k, F = c->info.ffn_mult * d;
    if (E == 1) {
        matmul(&y->up[0], c->h, c->ff, n);
        for (long i = 0; i < (long)n * F; i++) c->ff[i] = gelu(c->ff[i]);
        matmul(&y->down[0], c->ff, c->ffo, n);
        return;
    }
    /* route every byte: softmax over experts, keep the top k */
    float *w = c->tmp;  /* n * K weights */
    int *pick = c->idx; /* n * K experts */
    for (int t = 0; t < n; t++) {
        float *p = c->probs + (size_t)t * E, mx = -INFINITY, sum = 0.f;
        const float *ht = c->h + (size_t)t * d;
        for (int e = 0; e < E; e++) {
            float a = 0.f;
            for (int i = 0; i < d; i++) a += y->gate[(size_t)e * d + i] * ht[i];
            p[e] = a;
            if (a > mx) mx = a;
        }
        for (int e = 0; e < E; e++) sum += (p[e] = expf(p[e] - mx));
        for (int e = 0; e < E; e++) p[e] /= sum;
        float tot = 0.f;
        for (int j = 0; j < K; j++) {
            int best = -1;
            for (int e = 0; e < E; e++) {
                int used = 0;
                for (int u = 0; u < j; u++) used |= pick[t * K + u] == e;
                if (!used && (best < 0 || p[e] > p[best])) best = e;
            }
            pick[t * K + j] = best;
            tot += (w[t * K + j] = p[best]);
        }
        for (int j = 0; j < K; j++) w[t * K + j] = K == 1 ? 1.f : w[t * K + j] / tot;
    }
    memset(c->ffo, 0, sizeof(float) * n * d);
    /* run each expert once on the bytes routed to it */
    int *rows = c->idx + (size_t)n * K;
    for (int e = 0; e < E; e++) {
        int m = 0;
        for (int t = 0; t < n; t++)
            for (int j = 0; j < K; j++)
                if (pick[t * K + j] == e) rows[m++] = t;
        if (!m) continue;
        float *in = c->att, *out = c->qkvb;  /* free while the MoE runs */
        for (int i = 0; i < m; i++) memcpy(in + (size_t)i * d, c->h + (size_t)rows[i] * d, sizeof(float) * d);
        matmul(&y->up[e], in, c->ff, m);
        for (long i = 0; i < (long)m * F; i++) c->ff[i] = gelu(c->ff[i]);
        matmul(&y->down[e], c->ff, out, m);
        for (int i = 0; i < m; i++) {
            const int t = rows[i];
            float we = 0.f;
            for (int j = 0; j < K; j++)
                if (pick[t * K + j] == e) we = w[t * K + j];
            for (int q = 0; q < d; q++) c->ffo[(size_t)t * d + q] += we * out[(size_t)i * d + q];
        }
    }
}

int mlg_core_feed(mlg_core *c, const uint8_t *bytes, int n, float *logits, int all) {
    const int d = c->info.d_model, H = c->info.n_heads, hd = d / H, T = c->info.ctx;
    if (n <= 0) return c->n;
    if (c->n + n > T) return -1;
    const int p0 = c->n;
    memcpy(c->hist + p0, bytes, n);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < d; i++)
            c->x[(size_t)t * d + i] = c->tok[bytes[t] * d + i] + c->pos[(p0 + t) * d + i];
    const float scale = 1.f / sqrtf((float)hd);
    for (int l = 0; l < c->info.n_layers; l++) {
        const layer *y = &c->layers[l];
        float *K = c->k + (size_t)l * T * d, *V = c->v + (size_t)l * T * d;
        for (int t = 0; t < n; t++) layernorm(c->x + (size_t)t * d, y->ln1w, y->ln1b, c->h + (size_t)t * d, d);
        matmul(&y->qkv, c->h, c->qkvb, n);
        for (int t = 0; t < n; t++) {
            memcpy(K + (size_t)(p0 + t) * d, c->qkvb + (size_t)t * 3 * d + d, sizeof(float) * d);
            memcpy(V + (size_t)(p0 + t) * d, c->qkvb + (size_t)t * 3 * d + 2 * d, sizeof(float) * d);
        }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((long)n * (p0 + n) * d > (1L << 15))
#endif
        for (int t = 0; t < n; t++)
            for (int hh = 0; hh < H; hh++) {
                const int upto = p0 + t; /* causal: positions 0..upto */
                const float *q = c->qkvb + (size_t)t * 3 * d + hh * hd;
                float *out = c->att + (size_t)t * d + hh * hd;
                float mx = -INFINITY, sum = 0.f;
                float s[upto + 1];
                for (int j = 0; j <= upto; j++) {
                    const float *kj = K + (size_t)j * d + hh * hd;
                    float a = 0.f;
                    for (int i = 0; i < hd; i++) a += q[i] * kj[i];
                    s[j] = a * scale;
                    if (s[j] > mx) mx = s[j];
                }
                for (int j = 0; j <= upto; j++) sum += (s[j] = expf(s[j] - mx));
                for (int i = 0; i < hd; i++) out[i] = 0.f;
                for (int j = 0; j <= upto; j++) {
                    const float *vj = V + (size_t)j * d + hh * hd, wj = s[j] / sum;
                    for (int i = 0; i < hd; i++) out[i] += wj * vj[i];
                }
            }
        matmul(&y->proj, c->att, c->h, n);
        for (long i = 0; i < (long)n * d; i++) c->x[i] += c->h[i];
        for (int t = 0; t < n; t++) layernorm(c->x + (size_t)t * d, y->ln2w, y->ln2b, c->h + (size_t)t * d, d);
        moe(c, y, n);
        for (long i = 0; i < (long)n * d; i++) c->x[i] += c->ffo[i];
    }
    c->n += n;
    if (logits) {
        const int from = all ? 0 : n - 1;
        for (int t = from; t < n; t++) {
            float *h = c->h, *o = logits + (size_t)(t - from) * 256;
            layernorm(c->x + (size_t)t * d, c->lnw, c->lnb, h, d);
            for (int b = 0; b < 256; b++) {
                const float *e = c->tok + b * d;
                float a = 0.f;
                for (int i = 0; i < d; i++) a += e[i] * h[i];
                o[b] = a;
            }
        }
    }
    return c->n;
}

/* ---- generation ---------------------------------------------------------- */
static uint64_t rnd(uint64_t *s) {
    *s ^= *s >> 12, *s ^= *s << 25, *s ^= *s >> 27;
    return *s * 2685821657736338717ULL;
}

/* how many continuation bytes a UTF-8 lead byte announces; -1 if it cannot lead */
static int utf8_need(int b) {
    if (b < 0x80) return 0;
    if (b >= 0xC2 && b <= 0xDF) return 1;
    if (b >= 0xE0 && b <= 0xEF) return 2;
    if (b >= 0xF0 && b <= 0xF4) return 3;
    return -1;
}

static int sample(const float *logits, const mlg_sample *s, int need, uint64_t *rng) {
    float p[256];
    int order[256], allowed = 0;
    for (int b = 0; b < 256; b++) {
        int ok = 1;
        if (s->utf8) ok = need > 0 ? (b >= 0x80 && b <= 0xBF) : utf8_need(b) >= 0;
        p[b] = ok ? logits[b] : -INFINITY;
        allowed += ok;
    }
    if (!allowed) return '\n';
    int best = 0;
    for (int b = 1; b < 256; b++)
        if (p[b] > p[best]) best = b;
    if (s->temperature <= 0.f) return best;
    float sum = 0.f;
    for (int b = 0; b < 256; b++) sum += (p[b] = isinf(p[b]) ? 0.f : expf((p[b] - p[best]) / s->temperature));
    for (int b = 0; b < 256; b++) p[b] /= sum, order[b] = b;
    /* nucleus: the most likely bytes until top_p of the mass */
    for (int i = 1; i < 256; i++) {
        int o = order[i], j = i;
        while (j > 0 && p[order[j - 1]] < p[o]) order[j] = order[j - 1], j--;
        order[j] = o;
    }
    float keep = 0.f;
    int n = 0;
    while (n < 256 && (n == 0 || keep < s->top_p) && p[order[n]] > 0.f) keep += p[order[n++]];
    float u = (rnd(rng) >> 11) * (1.0f / 9007199254740992.0f) * keep, acc = 0.f;
    for (int i = 0; i < n; i++)
        if ((acc += p[order[i]]) >= u) return order[i];
    return order[n - 1];
}

int mlg_core_generate(mlg_core *c, const uint8_t *prompt, int n, uint8_t *out, int max_out,
                      const mlg_sample *s, mlg_on_byte cb, void *user) {
    const int T = c->info.ctx;
    float logits[256];
    uint64_t rng = s->seed ? s->seed : 0x9E3779B97F4A7C15ULL;
    if (n > T - 1) prompt += n - (T - 1), n = T - 1;
    mlg_core_reset(c);
    if (n > 0) mlg_core_feed(c, prompt, n, logits, 0);
    else {
        const uint8_t nl = '\n';
        mlg_core_feed(c, &nl, 1, logits, 0);
    }
    int produced = 0, need = 0;
    while (produced < max_out) {
        const int b = sample(logits, s, need, &rng);
        need = need > 0 ? need - 1 : (utf8_need(b) > 0 ? utf8_need(b) : 0);
        out[produced++] = (uint8_t)b;
        if (cb && cb((uint8_t)b, user)) break;
        if (s->stop_newline && b == '\n' && produced > s->min_bytes) break;
        const uint8_t byte = (uint8_t)b;
        if (c->n == T) {  /* the context is full: keep its second half and go on */
            uint8_t keep[T];
            const int k = T / 2;
            memcpy(keep, c->hist + T - k + 1, k - 1);
            keep[k - 1] = byte;
            mlg_core_reset(c);
            mlg_core_feed(c, keep, k, logits, 0);
        } else {
            mlg_core_feed(c, &byte, 1, logits, 0);
        }
    }
    return produced;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

void mlg_core_bench(mlg_core *c, double *prefill_ms, double *decode_ms) {
    const int T = c->info.ctx, n = T / 2, gen = T / 8 > 4 ? T / 8 : 4;
    uint8_t *text = malloc(n);
    float logits[256];
    for (int i = 0; i < n; i++) text[i] = "a maquina mede o proprio corpo "[i % 31];
    mlg_core_reset(c);
    double t = now_ms();
    mlg_core_feed(c, text, n, logits, 0);
    *prefill_ms = (now_ms() - t) / n;
    t = now_ms();
    for (int i = 0; i < gen; i++) {
        const uint8_t b = 'a' + i % 26;
        mlg_core_feed(c, &b, 1, logits, 0);
    }
    *decode_ms = (now_ms() - t) / gen;
    mlg_core_reset(c);
    free(text);
}

double mlg_core_bits_per_byte(mlg_core *c, const uint8_t *text, int N) {
    if (N < 2) return NAN;
    const int L = c->info.ctx < N - 1 ? c->info.ctx : N - 1, stride = L / 2 > 0 ? L / 2 : 1;
    float *logits = malloc(sizeof(float) * L * 256);
    double nats = 0.0;
    int done = 0;  /* bytes 1..done are scored */
    for (int s = 0;; s += stride) {
        if (s > N - 1 - L) s = N - 1 - L;
        mlg_core_reset(c);
        mlg_core_feed(c, text + s, L, logits, 1);
        for (int j = done - s; j < L; j++) {
            const float *o = logits + (size_t)j * 256;
            float mx = o[0];
            for (int b = 1; b < 256; b++) mx = o[b] > mx ? o[b] : mx;
            double z = 0.0;
            for (int b = 0; b < 256; b++) z += exp(o[b] - mx);
            nats += mx + log(z) - o[text[s + j + 1]];
        }
        done = s + L;
        if (s == N - 1 - L) break;
    }
    mlg_core_reset(c);
    free(logits);
    return nats / (N - 1) / log(2.0);
}

void mlg_set_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int mlg_get_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
