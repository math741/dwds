/* The trainable router: a linear map from hashed text features to intents,
 * trained from scratch in the lab (mlgsub/router.py) and read here.
 *
 * .mlgr layout (little endian):
 *   "MLGR" u32 version, u32 n_intents, u32 dim (a power of two)
 *   per intent: u32 name_len, name bytes
 *   f32 bias[n_intents], f32 w[n_intents][dim]
 *
 * Features of a text: the words of its normalized form, pairs of
 * neighbouring words, and the 3- and 4-byte pieces of " word word ",
 * each hashed into dim buckets; counts become log(1 + count), then the
 * vector is scaled to unit length.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlg.h"

struct mlg_router {
    uint32_t n, dim;
    char **names;
    float *bias, *w;
};

mlg_router *mlg_router_open(const char *path, char *err, int errlen) {
#define FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); if (fh) fclose(fh); mlg_router_close(r); return NULL; } while (0)
    mlg_router *r = calloc(1, sizeof *r);
    FILE *fh = fopen(path, "rb");
    uint32_t hdr[4];
    if (!r || !fh || fread(hdr, 4, 4, fh) != 4) FAIL("cannot read %s", path);
    if (memcmp(hdr, "MLGR", 4) || hdr[1] != 1 || hdr[2] == 0 || hdr[2] > 256 || hdr[3] == 0 ||
        (hdr[3] & (hdr[3] - 1)) || hdr[3] > (1u << 22))
        FAIL("%s is not an MLGR v1 router", path);
    r->n = hdr[2], r->dim = hdr[3];
    r->names = calloc(r->n, sizeof(char *));
    if (!r->names) FAIL("out of memory");
    for (uint32_t i = 0; i < r->n; i++) {
        uint32_t len;
        if (fread(&len, 4, 1, fh) != 1 || len > 255 || !(r->names[i] = calloc(len + 1, 1)) ||
            fread(r->names[i], 1, len, fh) != len)
            FAIL("truncated router");
    }
    r->bias = malloc(sizeof(float) * r->n);
    r->w = malloc(sizeof(float) * r->n * r->dim);
    if (!r->bias || !r->w || fread(r->bias, 4, r->n, fh) != r->n ||
        fread(r->w, 4, (size_t)r->n * r->dim, fh) != (size_t)r->n * r->dim)
        FAIL("truncated router");
    fclose(fh);
    return r;
#undef FAIL
}

void mlg_router_close(mlg_router *r) {
    if (!r) return;
    if (r->names)
        for (uint32_t i = 0; i < r->n; i++) free(r->names[i]);
    free(r->names), free(r->bias), free(r->w), free(r);
}

int mlg_router_count(const mlg_router *r) { return (int)r->n; }
const char *mlg_router_name(const mlg_router *r, int i) { return i >= 0 && (uint32_t)i < r->n ? r->names[i] : ""; }

static uint32_t kind_hash(int kind, const char *s, int n) {
    uint32_t h = 2166136261u ^ (uint32_t)(kind * 0x9E3779B1u);
    for (int i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 16777619u;
    return h;
}

/* Sparse features: bucket -> count, kept as a short list. */
typedef struct {
    uint32_t bucket[4096];
    float value[4096];
    int n;
} feats;

static void add(feats *f, uint32_t h, uint32_t dim) {
    const uint32_t b = h & (dim - 1);
    for (int i = 0; i < f->n; i++)
        if (f->bucket[i] == b) {
            f->value[i] += 1.f;
            return;
        }
    if (f->n < 4096) f->bucket[f->n] = b, f->value[f->n++] = 1.f;
}

int mlg_router_route(const mlg_router *r, const char *text, float *probs) {
    char s[1026];
    s[0] = ' ';
    int n = mlg_normalize(text, (int)strlen(text), s + 1, 1022) + 1;
    s[n++] = ' ';
    feats *f = calloc(1, sizeof *f);
    int prev = -1, prev_len = 0;
    for (int i = 1; i < n - 1;) {
        int j = i;
        while (j < n - 1 && s[j] != ' ') j++;
        if (j > i) {
            add(f, kind_hash(1, s + i, j - i), r->dim);
            if (prev >= 0) add(f, kind_hash(2, s + prev, prev_len + 1 + (j - i)), r->dim);
            prev = i, prev_len = j - i;
        }
        i = j + 1;
    }
    for (int g = 3; g <= 4; g++)
        for (int i = 0; i + g <= n; i++) add(f, kind_hash(g, s + i, g), r->dim);
    float norm = 0.f;
    for (int i = 0; i < f->n; i++) {
        f->value[i] = log1pf(f->value[i]);
        norm += f->value[i] * f->value[i];
    }
    norm = norm > 0.f ? 1.f / sqrtf(norm) : 0.f;
    float mx = -INFINITY, sum = 0.f;
    int best = 0;
    for (uint32_t c = 0; c < r->n; c++) {
        float a = r->bias[c];
        const float *w = r->w + (size_t)c * r->dim;
        for (int i = 0; i < f->n; i++) a += w[f->bucket[i]] * f->value[i] * norm;
        probs[c] = a;
        if (a > mx) mx = a, best = (int)c;
    }
    for (uint32_t c = 0; c < r->n; c++) sum += (probs[c] = expf(probs[c] - mx));
    for (uint32_t c = 0; c < r->n; c++) probs[c] /= sum;
    free(f);
    return best;
}
