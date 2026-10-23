/* Builds the retrieval index (.mlgk, see mlg_know.c) from passages, one per
 * line as "title<TAB>text". Runs in the lab (mlgsub/know.py calls it), in C so
 * that it tokenizes exactly like the headset and gets through Wikipedia fast.
 * Words in more than max_df of the passages carry almost no evidence and are
 * left out, which keeps the index a fraction of the text's size. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlg.h"

typedef struct {
    uint32_t hash, doc;
    uint16_t tf;
} post;

typedef struct {
    uint64_t off;
    uint32_t title_len, text_len, dl, pad;
} doc_rec;

static int by_hash_doc(const void *a, const void *b) {
    const post *x = a, *y = b;
    if (x->hash != y->hash) return x->hash < y->hash ? -1 : 1;
    return x->doc < y->doc ? -1 : x->doc > y->doc;
}

static int by_hash(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

#define GROW(ptr, n, cap) do { if ((n) >= (cap)) { (cap) = (cap) ? 2 * (cap) : 1024; \
    (ptr) = realloc((ptr), sizeof *(ptr) * (cap)); if (!(ptr)) goto oom; } } while (0)

int mlg_know_build(const char *tsv, const char *out_path, float max_df, char *err, int errlen) {
    FILE *in = fopen(tsv, "rb"), *out = NULL;
    if (!in) return snprintf(err, errlen, "cannot read %s", tsv), -1;
    post *posts = NULL;
    doc_rec *docs = NULL;
    char *strings = NULL, *line = NULL, *norm = NULL;
    size_t n_posts = 0, cap_posts = 0, n_docs = 0, cap_docs = 0, n_str = 0, cap_str = 0, cap_line = 0;
    double total_dl = 0;
    uint32_t *hs = NULL;
    size_t cap_hs = 0;
    ssize_t len;
    while ((len = getline(&line, &cap_line, in)) > 0) {
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) len--;
        char *tab = memchr(line, '\t', len);
        if (!tab) continue;
        const int tl = (int)(tab - line), xl = (int)(len - tl - 1);
        if (xl <= 0) continue;
        GROW(docs, n_docs, cap_docs);
        while (n_str + len >= cap_str) {
            cap_str = cap_str ? 2 * cap_str : 1 << 20;
            if (!(strings = realloc(strings, cap_str))) goto oom;
        }
        docs[n_docs] = (doc_rec){n_str, (uint32_t)tl, (uint32_t)xl, 0, 0};
        memcpy(strings + n_str, line, tl);
        memcpy(strings + n_str + tl, tab + 1, xl);
        n_str += tl + xl;
        /* the title counts twice: it says what the passage is about */
        size_t n_hs = 0;
        for (int part = 0; part < 3; part++) {
            const char *src = part < 2 ? line : tab + 1;
            const int sl = part < 2 ? tl : xl;
            if (!(norm = realloc(norm, 2 * sl + 8))) goto oom;
            const int n = mlg_normalize(src, sl, norm, 2 * sl + 8);
            for (int i = 0; i < n;) {
                int j = i;
                while (j < n && norm[j] != ' ') j++;
                if (j - i >= 2) {
                    GROW(hs, n_hs, cap_hs);
                    hs[n_hs++] = mlg_hash(norm + i, j - i);
                }
                i = j + 1;
            }
        }
        docs[n_docs].dl = (uint32_t)n_hs;
        total_dl += n_hs;
        qsort(hs, n_hs, sizeof *hs, by_hash);
        for (size_t i = 0; i < n_hs;) {
            size_t j = i;
            while (j < n_hs && hs[j] == hs[i]) j++;
            GROW(posts, n_posts, cap_posts);
            posts[n_posts++] = (post){hs[i], (uint32_t)n_docs, (uint16_t)(j - i > 65535 ? 65535 : j - i)};
            i = j;
        }
        n_docs++;
    }
    fclose(in);
    in = NULL;
    if (!n_docs) return snprintf(err, errlen, "no passages in %s", tsv), -1;
    qsort(posts, n_posts, sizeof *posts, by_hash_doc);
    /* terms: hash, df, offset into postings; very common words are dropped */
    /* a small index keeps every word; only a large one can afford to drop the common ones */
    const size_t limit = (size_t)(max_df * n_docs) > 2000 ? (size_t)(max_df * n_docs) : 2000;
    out = fopen(out_path, "wb");
    if (!out) return snprintf(err, errlen, "cannot write %s", out_path), -1;
    size_t n_terms = 0, kept = 0;
    for (size_t i = 0; i < n_posts;) {
        size_t j = i;
        while (j < n_posts && posts[j].hash == posts[i].hash) j++;
        if (j - i <= limit) n_terms++, kept += j - i;
        i = j;
    }
    const uint64_t docs_off = 64, terms_off = docs_off + n_docs * sizeof(doc_rec),
                   postings_off = terms_off + n_terms * 16, strings_off = postings_off + kept * 6;
    const float avgdl = (float)(total_dl / n_docs), k1 = 1.2f, b = 0.75f;
    uint32_t hdr[8] = {0, 1, (uint32_t)n_docs, (uint32_t)n_terms, 0, 0, 0, 0};
    memcpy(hdr, "MLGK", 4);
    memcpy(&hdr[4], &avgdl, 4), memcpy(&hdr[5], &k1, 4), memcpy(&hdr[6], &b, 4);
    const uint64_t offs[4] = {docs_off, terms_off, postings_off, strings_off};
    fwrite(hdr, 4, 8, out);
    fwrite(offs, 8, 4, out);
    fwrite(docs, sizeof(doc_rec), n_docs, out);
    uint64_t at = 0;
    for (size_t i = 0; i < n_posts;) {
        size_t j = i;
        while (j < n_posts && posts[j].hash == posts[i].hash) j++;
        if (j - i <= limit) {
            const uint32_t rec[2] = {posts[i].hash, (uint32_t)(j - i)};
            fwrite(rec, 4, 2, out);
            fwrite(&at, 8, 1, out);
            at += (j - i) * 6;
        }
        i = j;
    }
    for (size_t i = 0; i < n_posts;) {
        size_t j = i;
        while (j < n_posts && posts[j].hash == posts[i].hash) j++;
        if (j - i <= limit) {
            for (size_t q = i; q < j; q++) fwrite(&posts[q].doc, 4, 1, out);
            for (size_t q = i; q < j; q++) fwrite(&posts[q].tf, 2, 1, out);
        }
        i = j;
    }
    fwrite(strings, 1, n_str, out);
    fclose(out);
    snprintf(err, errlen, "%zu passages, %zu terms kept of %zu postings", n_docs, n_terms, n_posts);
    free(posts), free(docs), free(strings), free(line), free(norm), free(hs);
    return (int)n_docs;
oom:
    if (in) fclose(in);
    free(posts), free(docs), free(strings), free(line), free(norm), free(hs);
    return snprintf(err, errlen, "out of memory"), -1;
}
