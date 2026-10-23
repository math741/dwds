/* Desktop tool for the headset engine: the same C the Quest runs.
 *
 *   mlg_cli info   brain.mlgb
 *   mlg_cli logits brain.mlgb input.bin out.f32   every byte's logits (parity with PyTorch)
 *   mlg_cli bpb    brain.mlgb text.txt            bits per byte, scored like the lab
 *   mlg_cli say    brain.mlgb "U: oi\nM:" [temperature [utf8]]   generate after a prompt
 *   mlg_cli bench  brain.mlgb [threads]
 *   mlg_cli index  passages.tsv index.mlgk [max_df]   build the retrieval index
 *   mlg_cli search index.mlgk "query" [k]
 *   mlg_cli batch  index.mlgk k < queries          best k passage ids per query line
 *   mlg_cli route  router.mlgr "text"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlg.h"

static unsigned char *slurp(const char *path, long *n) {
    FILE *fh = fopen(path, "rb");
    if (!fh) return NULL;
    fseek(fh, 0, SEEK_END);
    *n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    unsigned char *b = malloc(*n + 1);
    if (fread(b, 1, *n, fh) != (size_t)*n) *n = 0;
    fclose(fh);
    return b;
}

/* "\n" in a command-line prompt means a newline */
static int unescape(const char *in, unsigned char *out) {
    int n = 0;
    for (; *in; in++) {
        if (in[0] == '\\' && in[1] == 'n') out[n++] = '\n', in++;
        else out[n++] = (unsigned char)*in;
    }
    return n;
}

static int print_byte(uint8_t b, void *user) {
    (void)user;
    fputc(b, stdout);
    fflush(stdout);
    return 0;
}

static mlg_core *open_core(const char *path) {
    char err[256];
    mlg_core *c = mlg_core_open(path, err, sizeof err);
    if (!c) fprintf(stderr, "%s\n", err), exit(1);
    return c;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: mlg_cli info|logits|bpb|say|bench|search|route FILE ...\n");
        return 2;
    }
    const char *cmd = argv[1];
    char err[256];
    if (!strcmp(cmd, "info")) {
        mlg_core *c = open_core(argv[2]);
        mlg_core_info i = mlg_core_get_info(c);
        printf("d_model %d  layers %d  heads %d  experts %d top-%d  ffn x%d  ctx %d  %s  %.2fM params\n",
               i.d_model, i.n_layers, i.n_heads, i.n_experts, i.top_k, i.ffn_mult, i.ctx,
               i.wtype ? "q8" : "f32", i.params / 1e6);
        mlg_core_close(c);
    } else if (!strcmp(cmd, "logits") && argc >= 5) {
        mlg_core *c = open_core(argv[2]);
        long n;
        unsigned char *in = slurp(argv[3], &n);
        if (!in || n <= 0 || n > mlg_core_get_info(c).ctx) return fprintf(stderr, "bad input\n"), 1;
        float *lg = malloc(sizeof(float) * 256 * n);
        int half = (int)n / 2;  /* in two feeds, so the cached path is checked too */
        mlg_core_feed(c, in, half, lg, 1);
        mlg_core_feed(c, in + half, (int)n - half, lg + 256 * half, 1);
        FILE *fh = fopen(argv[4], "wb");
        fwrite(lg, sizeof(float), 256 * n, fh);
        fclose(fh);
        mlg_core_close(c);
    } else if (!strcmp(cmd, "bpb") && argc >= 4) {
        mlg_core *c = open_core(argv[2]);
        long n;
        unsigned char *t = slurp(argv[3], &n);
        if (!t) return fprintf(stderr, "cannot read %s\n", argv[3]), 1;
        printf("%.4f\n", mlg_core_bits_per_byte(c, t, (int)n));
        mlg_core_close(c);
    } else if (!strcmp(cmd, "say") && argc >= 4) {
        mlg_core *c = open_core(argv[2]);
        unsigned char prompt[8192], out[4096];
        int n = unescape(argv[3], prompt);
        mlg_sample s = {.temperature = 0.6f, .top_p = 0.9f, .seed = 7, .stop_newline = 1,
                        .min_bytes = 4, .utf8 = 1};
        if (argc >= 5) s.temperature = (float)atof(argv[4]);
        if (argc >= 6) s.utf8 = atoi(argv[5]), s.stop_newline = s.utf8;
        mlg_core_generate(c, prompt, n, out, sizeof out, &s, print_byte, NULL);
        printf("\n");
        mlg_core_close(c);
    } else if (!strcmp(cmd, "bench")) {
        mlg_core *c = open_core(argv[2]);
        if (argc >= 4) mlg_set_threads(atoi(argv[3]));
        double pre, dec;
        mlg_core_bench(c, &pre, &dec);
        printf("threads %d  prefill %.3f ms/byte  decode %.3f ms/byte (%.0f bytes/s)\n",
               mlg_get_threads(), pre, dec, 1000.0 / dec);
        mlg_core_close(c);
    } else if (!strcmp(cmd, "index") && argc >= 4) {
        const int n = mlg_know_build(argv[2], argv[3], argc >= 5 ? (float)atof(argv[4]) : 0.05f, err, sizeof err);
        printf("%s\n", err);
        return n < 0;
    } else if (!strcmp(cmd, "search") && argc >= 4) {
        mlg_know *k = mlg_know_open(argv[2], err, sizeof err);
        if (!k) return fprintf(stderr, "%s\n", err), 1;
        int top = argc >= 5 ? atoi(argv[4]) : 3, ids[64], len, tl;
        float sc[64];
        if (top > 64) top = 64;
        const int n = mlg_know_search(k, argv[3], top, ids, sc);
        for (int i = 0; i < n; i++) {
            const char *title = mlg_know_title(k, ids[i], &tl), *text = mlg_know_text(k, ids[i], &len);
            printf("[%.2f] %.*s | %.*s\n", sc[i], tl, title, len > 300 ? 300 : len, text);
        }
        mlg_know_close(k);
    } else if (!strcmp(cmd, "batch") && argc >= 4) {
        mlg_know *k = mlg_know_open(argv[2], err, sizeof err);
        if (!k) return fprintf(stderr, "%s\n", err), 1;
        int top = atoi(argv[3]), ids[64];
        float sc[64];
        if (top < 1 || top > 64) top = 64;
        char *line = NULL;
        size_t cap = 0;
        ssize_t len;
        while ((len = getline(&line, &cap, stdin)) > 0) {
            if (line[len - 1] == '\n') line[len - 1] = 0;
            const int n = mlg_know_search(k, line, top, ids, sc);
            for (int i = 0; i < n; i++) printf(i ? "\t%d" : "%d", ids[i]);
            printf("\n");
        }
        free(line);
        mlg_know_close(k);
    } else if (!strcmp(cmd, "route") && argc >= 4) {
        mlg_router *r = mlg_router_open(argv[2], err, sizeof err);
        if (!r) return fprintf(stderr, "%s\n", err), 1;
        float p[256];
        const int best = mlg_router_route(r, argv[3], p);
        for (int i = 0; i < mlg_router_count(r); i++)
            printf("%s%-10s %.3f\n", i == best ? "> " : "  ", mlg_router_name(r, i), p[i]);
        mlg_router_close(r);
    } else {
        fprintf(stderr, "unknown command %s\n", cmd);
        return 2;
    }
    return 0;
}
