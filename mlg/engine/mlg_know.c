/* The retrieval expert: what the MLG knows, searched with BM25.
 *
 * .mlgk layout (little endian), written by mlgsub/know.py:
 *   "MLGK" u32 version, u32 n_docs, u32 n_terms, f32 avgdl, f32 k1, f32 b, u32 pad
 *   u64 docs_off, terms_off, postings_off, strings_off
 *   docs:     n_docs  x {u64 off, u32 title_len, u32 text_len, u32 dl, u32 pad}
 *   terms:    n_terms x {u32 hash, u32 df, u64 off}, sorted by hash
 *   postings: per term, u32 doc[df] then u16 tf[df]
 *   strings:  title bytes followed by text bytes, at each doc's off
 * The file is mapped, not read: a large index costs disk, not memory.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__wasi__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "mlg.h"

typedef struct {
    uint64_t off;
    uint32_t title_len, text_len, dl, pad;
} doc_rec;

typedef struct {
    uint32_t hash, df;
    uint64_t off;
} term_rec;

struct mlg_know {
    unsigned char *map;
    size_t size;
    int owned;        /* map is a heap copy (browser), not a mapping */
    uint32_t n_docs, n_terms;
    float avgdl, k1, b;
    const doc_rec *docs;
    const term_rec *terms;
    const unsigned char *postings, *strings;
    float *score;     /* per doc, zero between searches */
    uint32_t *touched;
    uint32_t *titles; /* open addressing: lead passage id + 1 of each article, by its title */
    uint32_t n_titles;
    uint32_t *asks;   /* the same, for every question of the MLG's own "§ q1 | q2" passages */
    uint32_t n_asks;
};

/* An article's title as asked for: normalized, without its "(disambiguation)". */
static int title_key(const char *title, int len, char *out, int cap) {
    for (int i = 0; i + 1 < len; i++)
        if (title[i] == ' ' && title[i + 1] == '(') { len = i; break; }
    const int n = mlg_normalize(title, len, out, cap - 1);
    out[n] = 0;
    return n;
}

static int same_title(const mlg_know *k, uint32_t a, uint32_t b) {
    return k->docs[a].title_len == k->docs[b].title_len &&
           !memcmp(k->strings + k->docs[a].off, k->strings + k->docs[b].off, k->docs[a].title_len);
}

static int is_own(const char *t, int tl) { return tl >= 2 && (unsigned char)t[0] == 0xC2 && (unsigned char)t[1] == 0xA7; }

/* the i-th question of an own passage's title "§ q1 | q2 | ...", normalized; 0 when there is none */
static int own_question(const mlg_know *k, uint32_t id, int i, char *out, int cap) {
    const char *t = (const char *)k->strings + k->docs[id].off;
    const int tl = (int)k->docs[id].title_len;
    int a = 2;
    for (int q = 0; q < i && a < tl; a++)
        if (a + 2 < tl && t[a] == ' ' && t[a + 1] == '|' && t[a + 2] == ' ') q++, a += 2;
    if (a >= tl) return 0;
    int b = a;
    while (b < tl && !(b + 2 < tl && t[b] == ' ' && t[b + 1] == '|' && t[b + 2] == ' ')) b++;
    const int n = mlg_normalize(t + a, b - a, out, cap - 1);
    out[n] = 0;
    return n;
}

/* the MLG's own questions, word for word: "o que é a mlg" -> its answer, however large the index */
static int index_asks(mlg_know *k) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < k->n_docs; i++) {
        const char *t = (const char *)k->strings + k->docs[i].off;
        const int tl = (int)k->docs[i].title_len;
        if (is_own(t, tl))
            for (int c = 2; c + 2 < tl; c++) n += t[c] == ' ' && t[c + 1] == '|' && t[c + 2] == ' ';
        n += is_own(t, tl);
    }
    uint32_t cap = 16;
    while (cap < 2 * n) cap <<= 1;
    k->asks = calloc(cap, sizeof(uint32_t));
    if (!k->asks) return 0;
    k->n_asks = cap;
    char key[512], other[512];
    for (uint32_t i = 0; i < k->n_docs; i++) {
        if (!is_own((const char *)k->strings + k->docs[i].off, (int)k->docs[i].title_len)) continue;
        for (int q = 0;; q++) {
            const int len = own_question(k, i, q, key, sizeof key);
            if (len <= 0) break;
            uint32_t h = mlg_hash(key, len) & (cap - 1), dup = 0;
            for (; k->asks[h]; h = (h + 1) & (cap - 1)) {
                const uint32_t o = k->asks[h] - 1;
                for (int j = 0; !dup; j++) {  /* the first passage asked this way wins */
                    const int ol = own_question(k, o, j, other, sizeof other);
                    if (ol <= 0) break;
                    dup = ol == len && !memcmp(other, key, len);
                }
                if (dup) break;
            }
            if (!dup) k->asks[h] = i + 1;
        }
    }
    return 1;
}

/* the first passage of every article (its lead), found by title in O(1) */
static int index_titles(mlg_know *k) {
    uint32_t leads = 0;
    for (uint32_t i = 0; i < k->n_docs; i++) leads += !i || !same_title(k, i, i - 1);
    uint32_t cap = 16;
    while (cap < 2 * leads) cap <<= 1;
    k->titles = calloc(cap, sizeof(uint32_t));
    if (!k->titles) return 0;
    k->n_titles = cap;
    char key[512];
    for (uint32_t i = 0; i < k->n_docs; i++) {
        if (i && same_title(k, i, i - 1)) continue;
        const char *t = (const char *)k->strings + k->docs[i].off;
        const int tl = (int)k->docs[i].title_len;
        if (is_own(t, tl)) continue;  /* the MLG's own "§" passages are found by question (index_asks) */
        const int n = title_key(t, tl, key, sizeof key);
        if (n <= 0) continue;
        uint32_t h = mlg_hash(key, n) & (cap - 1);
        while (k->titles[h]) {
            char other[512];
            const uint32_t o = k->titles[h] - 1;
            if (title_key((const char *)k->strings + k->docs[o].off, (int)k->docs[o].title_len, other, sizeof other) == n &&
                !memcmp(other, key, n)) break;  /* the first article of a title wins */
            h = (h + 1) & (cap - 1);
        }
        if (!k->titles[h]) k->titles[h] = i + 1;
    }
    return 1;
}

static mlg_know *know_parse(mlg_know *k, char *err, int errlen) {
#define FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); mlg_know_close(k); return NULL; } while (0)
    const unsigned char *p = k->map;
    uint32_t u[8];
    uint64_t off[4];
    if (k->size < 64) FAIL("truncated index");
    memcpy(u, p, 32);
    memcpy(off, p + 32, 32);
    if (memcmp(p, "MLGK", 4) || u[1] != 1) FAIL("not an MLGK v1 index");
    k->n_docs = u[2], k->n_terms = u[3];
    memcpy(&k->avgdl, &u[4], 4), memcpy(&k->k1, &u[5], 4), memcpy(&k->b, &u[6], 4);
    if (off[0] + (uint64_t)k->n_docs * sizeof(doc_rec) > k->size ||
        off[1] + (uint64_t)k->n_terms * sizeof(term_rec) > k->size || off[2] > k->size || off[3] > k->size)
        FAIL("truncated index");
    k->docs = (const doc_rec *)(p + off[0]);
    k->terms = (const term_rec *)(p + off[1]);
    k->postings = p + off[2];
    k->strings = p + off[3];
    k->score = calloc(k->n_docs ? k->n_docs : 1, sizeof(float));
    k->touched = malloc(sizeof(uint32_t) * (k->n_docs ? k->n_docs : 1));
    if (!k->score || !k->touched || !index_titles(k) || !index_asks(k)) FAIL("out of memory");
    return k;
#undef FAIL
}

/* the browser has no files to map: the index arrives as bytes (owned from now on) */
mlg_know *mlg_know_open_mem(unsigned char *bytes, long size, char *err, int errlen) {
    mlg_know *k = calloc(1, sizeof *k);
    if (!k) { free(bytes); return NULL; }
    k->map = bytes, k->size = (size_t)size, k->owned = 1;
    return know_parse(k, err, errlen);
}

mlg_know *mlg_know_open(const char *path, char *err, int errlen) {
#if defined(__wasi__)
    FILE *fh = fopen(path, "rb");
    if (!fh) { if (err) snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    unsigned char *b = malloc(size > 0 ? size : 1);
    const long got = b ? (long)fread(b, 1, size, fh) : 0;
    fclose(fh);
    if (got != size) { free(b); if (err) snprintf(err, errlen, "cannot read %s", path); return NULL; }
    return mlg_know_open_mem(b, size, err, errlen);
#else
    mlg_know *k = calloc(1, sizeof *k);
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (!k || fd < 0 || fstat(fd, &st) || st.st_size < 64) {
        if (fd >= 0) close(fd);
        if (err) snprintf(err, errlen, "cannot open %s", path);
        free(k);
        return NULL;
    }
    k->size = st.st_size;
    k->map = mmap(NULL, k->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (k->map == MAP_FAILED) {
        k->map = NULL;
        if (err) snprintf(err, errlen, "cannot map %s", path);
        free(k);
        return NULL;
    }
    return know_parse(k, err, errlen);
#endif
}

void mlg_know_close(mlg_know *k) {
    if (!k) return;
#if !defined(__wasi__)
    if (k->map && !k->owned) munmap(k->map, k->size);
#endif
    if (k->owned) free(k->map);
    free(k->score), free(k->touched), free(k->titles), free(k->asks), free(k);
}

int mlg_know_count(const mlg_know *k) { return (int)k->n_docs; }

static const term_rec *find(const mlg_know *k, uint32_t h) {
    uint32_t lo = 0, hi = k->n_terms;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (k->terms[mid].hash < h) lo = mid + 1;
        else hi = mid;
    }
    return lo < k->n_terms && k->terms[lo].hash == h ? &k->terms[lo] : NULL;
}

int mlg_know_search(const mlg_know *k, const char *query, int top, int *ids, float *scores) {
    char norm[2048];
    const int n = mlg_normalize(query, (int)strlen(query), norm, sizeof norm);
    uint32_t seen[128];
    int n_seen = 0, n_touched = 0;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && norm[j] != ' ') j++;
        if (j - i >= 2 && n_seen < 128) {
            const uint32_t h = mlg_hash(norm + i, j - i);
            int dup = 0;
            for (int s = 0; s < n_seen; s++) dup |= seen[s] == h;
            const term_rec *t = dup ? NULL : find(k, h);
            if (t) {
                seen[n_seen++] = h;
                const float idf = logf(1.f + (k->n_docs - t->df + 0.5f) / (t->df + 0.5f));
                const unsigned char *base = k->postings + t->off;
                for (uint32_t q = 0; q < t->df; q++) {
                    uint32_t d;
                    uint16_t tf;
                    memcpy(&d, base + 4 * q, 4);
                    memcpy(&tf, base + 4 * t->df + 2 * q, 2);
                    if (d >= k->n_docs) continue;
                    const float norm_len = k->k1 * (1.f - k->b + k->b * k->docs[d].dl / k->avgdl);
                    if (k->score[d] == 0.f) k->touched[n_touched++] = d;
                    k->score[d] += idf * tf * (k->k1 + 1.f) / (tf + norm_len);
                }
            }
        }
        i = j + 1;
    }
    int found = 0;
    for (int i = 0; i < n_touched; i++) {
        const uint32_t d = k->touched[i];
        const float s = k->score[d];
        if (found < top || s > scores[found - 1]) {
            int j = found < top ? found++ : top - 1;
            while (j > 0 && scores[j - 1] < s) ids[j] = ids[j - 1], scores[j] = scores[j - 1], j--;
            ids[j] = (int)d, scores[j] = s;
        }
        k->score[d] = 0.f;
    }
    return found;
}

int mlg_know_lookup(const mlg_know *k, const char *key, int n) {
    if (!k->n_titles || n <= 0) return -1;
    uint32_t h = mlg_hash(key, n) & (k->n_titles - 1);
    char other[512];
    for (; k->titles[h]; h = (h + 1) & (k->n_titles - 1)) {
        const uint32_t o = k->titles[h] - 1;
        if (title_key((const char *)k->strings + k->docs[o].off, (int)k->docs[o].title_len, other, sizeof other) == n &&
            !memcmp(other, key, n)) return (int)o;
    }
    return -1;
}

int mlg_know_asked(const mlg_know *k, const char *key, int n) {
    if (!k->n_asks || n <= 0) return -1;
    char other[512];
    for (uint32_t h = mlg_hash(key, n) & (k->n_asks - 1); k->asks[h]; h = (h + 1) & (k->n_asks - 1)) {
        const uint32_t o = k->asks[h] - 1;
        for (int j = 0;; j++) {
            const int ol = own_question(k, o, j, other, sizeof other);
            if (ol <= 0) break;
            if (ol == n && !memcmp(other, key, n)) return (int)o;
        }
    }
    return -1;
}

int mlg_know_lead(const mlg_know *k, int id) {
    if (id < 0 || (uint32_t)id >= k->n_docs) return -1;
    while (id > 0 && same_title(k, (uint32_t)id, (uint32_t)id - 1)) id--;
    return id;
}

float mlg_know_idf(const mlg_know *k, const char *word, int n) {
    const term_rec *t = find(k, mlg_hash(word, n));
    return t ? logf(1.f + (k->n_docs - t->df + 0.5f) / (t->df + 0.5f)) : -1.f;
}

const char *mlg_know_title(const mlg_know *k, int id, int *len) {
    if (id < 0 || (uint32_t)id >= k->n_docs) return *len = 0, "";
    *len = (int)k->docs[id].title_len;
    return (const char *)k->strings + k->docs[id].off;
}

const char *mlg_know_text(const mlg_know *k, int id, int *len) {
    if (id < 0 || (uint32_t)id >= k->n_docs) return *len = 0, "";
    *len = (int)k->docs[id].text_len;
    return (const char *)k->strings + k->docs[id].off + k->docs[id].title_len;
}
