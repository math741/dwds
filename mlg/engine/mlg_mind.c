/* The MLG's mind: everything between hearing a sentence and answering it.
 *
 * The headset (through JNI) and the lab (through ctypes, mlgsub/mind.py) run
 * this same code, so the lab can put the whole MLG through its behaviour suite
 * (data/trust/behaviour.tsv) before anything reaches the Quest.
 *
 *   think(text) -> intent (router) -> organ:
 *     social      greetings, thanks, goodbyes: fixed replies from the corpus
 *     procedural  arithmetic, units, date and time: exact
 *     memory      what the user asked to keep, and recalling it
 *     body, lab   facts the host reports (sensors, the live link)
 *     knowledge   a question the MLG's own corpus answers word for word
 *     core        everything else: a prompt with retrieved evidence and
 *                 remembered facts, for a Resident Core (the lab's or this body's)
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mlg.h"

#define MAX_FACTS 512
#define FACT_LEN 400

typedef struct {
    long long at;
    char key[48];
    char text[FACT_LEN];
} fact;

struct mlg_mind {
    mlg_router *router;
    mlg_know *know;
    mlg_core *core;
    char memory_path[512];
    fact facts[MAX_FACTS];
    int n_facts;
    char body[2048];   /* key=value lines from the host */
    char lab[512];     /* the live link's state, as a sentence */
    long long now;     /* local time from the host */
    char spoke[16];    /* who gave the last answer: "core", "knowledge" (evidence), "trust" (no one) */
    int tz_minutes;
    unsigned turn;
};

static void json_escape(const char *s, char *out, int cap);
static int has_digit(const char *w);

/* ---- small text helpers ----------------------------------------------------- */
static int norm(const char *s, char *out, int cap) {
    const int n = mlg_normalize(s, (int)strlen(s), out, cap - 1);
    out[n] = 0;
    return n;
}

/* split normalized text into words (pointers into buf, NUL-terminated in place) */
static int words(char *buf, char **w, int max) {
    int n = 0;
    for (char *p = strtok(buf, " "); p && n < max; p = strtok(NULL, " ")) w[n++] = p;
    return n;
}

static int starts(const char *s, const char *prefix) { return strncmp(s, prefix, strlen(prefix)) == 0; }

static int has_word(char **w, int n, const char *x) {
    for (int i = 0; i < n; i++)
        if (!strcmp(w[i], x)) return 1;
    return 0;
}

static void copy(char *dst, int cap, const char *src) {
    snprintf(dst, cap, "%s", src);
}

static void trim(char *s) {
    char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    memmove(s, p, strlen(p) + 1);
    int n = (int)strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

/* lowercase with accents folded, but punctuation and symbols kept (for arithmetic) */
static void fold(const char *in, char *out, int cap) {
    static const char L1[] = "aaaaaaaceeeeiiiidnooooo ouuuuy saaaaaaaceeeeiiiidnooooo ouuuuy y";
    const unsigned char *s = (const unsigned char *)in;
    int o = 0;
    for (int i = 0; s[i] && o < cap - 1; i++) {
        if (s[i] == 0xC3 && s[i + 1] >= 0x80 && s[i + 1] <= 0xBF) out[o++] = L1[s[++i] - 0x80];
        else if (s[i] == 0xC3 && s[i + 1]) i++;
        else if (s[i] == 0xC2 && s[i + 1]) { i++; out[o++] = ' '; }
        else if (s[i] == 0xC3 || s[i] >= 0x80) out[o++] = ' ';
        else out[o++] = (char)tolower(s[i]);
    }
    out[o] = 0;
}

/* ---- memory ----------------------------------------------------------------- */
static void json_str(FILE *fh, const char *s) {
    fputc('"', fh);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', fh), fputc(*s, fh);
        else if ((unsigned char)*s < 0x20) fprintf(fh, "\\u%04x", *s);
        else fputc(*s, fh);
    }
    fputc('"', fh);
}

/* reads "key":"value" from one JSON line written by save_fact (no general JSON) */
static void field(const char *line, const char *key, char *out, int cap) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(line, pat);
    out[0] = 0;
    if (!p) return;
    p += strlen(pat);
    if (*p == '"') {
        int o = 0;
        for (p++; *p && *p != '"' && o < cap - 1; p++) {
            if (*p == '\\' && p[1]) {
                p++;
                if (*p == 'u') { out[o++] = ' '; p += 4; continue; }
            }
            out[o++] = *p;
        }
        out[o] = 0;
    } else {
        int o = 0;
        while (*p && *p != ',' && *p != '}' && o < cap - 1) out[o++] = *p++;
        out[o] = 0;
    }
}

static void add_fact(mlg_mind *m, long long at, const char *key, const char *text) {
    if (key[0])  /* a newer "meu nome é" replaces the older one */
        for (int i = 0; i < m->n_facts; i++)
            if (!strcmp(m->facts[i].key, key)) {
                memmove(&m->facts[i], &m->facts[i + 1], sizeof(fact) * (m->n_facts - i - 1));
                m->n_facts--, i--;
            }
    if (m->n_facts == MAX_FACTS) memmove(&m->facts[0], &m->facts[1], sizeof(fact) * --m->n_facts);
    fact *f = &m->facts[m->n_facts++];
    f->at = at;
    copy(f->key, sizeof f->key, key);
    copy(f->text, sizeof f->text, text);
}

mlg_mind *mlg_mind_new(const char *memory_path) {
    mlg_mind *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    if (memory_path) copy(m->memory_path, sizeof m->memory_path, memory_path);
    FILE *fh = memory_path && memory_path[0] ? fopen(memory_path, "r") : NULL;
    if (fh) {
        char line[1024], key[48], text[FACT_LEN], at[32];
        while (fgets(line, sizeof line, fh)) {
            field(line, "k", key, sizeof key);
            field(line, "v", text, sizeof text);
            field(line, "t", at, sizeof at);
            if (text[0]) add_fact(m, atoll(at), key, text);
        }
        fclose(fh);
    }
    m->now = (long long)time(NULL);
    return m;
}

void mlg_mind_free(mlg_mind *m) { free(m); }

void mlg_mind_organs(mlg_mind *m, mlg_router *r, mlg_know *k, mlg_core *c) { m->router = r, m->know = k, m->core = c; }
void mlg_mind_body(mlg_mind *m, const char *facts) { copy(m->body, sizeof m->body, facts ? facts : ""); }
void mlg_mind_lab(mlg_mind *m, const char *status) { copy(m->lab, sizeof m->lab, status ? status : ""); }
void mlg_mind_clock(mlg_mind *m, long long unix_seconds, int tz_minutes) { m->now = unix_seconds, m->tz_minutes = tz_minutes; }

int mlg_mind_memory_count(const mlg_mind *m) { return m->n_facts; }

const char *mlg_mind_spoke(const mlg_mind *m) { return m->spoke; }

/* the memory as JSON lines (what the memory file holds), for hosts without files */
int mlg_mind_memory_dump(const mlg_mind *m, char *out, int cap) {
    int o = 0;
    for (int i = 0; i < m->n_facts && o < cap; i++) {
        char k[100], v[FACT_LEN * 2];
        json_escape(m->facts[i].key, k, sizeof k);
        json_escape(m->facts[i].text, v, sizeof v);
        o += snprintf(out + o, cap - o, "{\"t\":%lld,\"k\":\"%s\",\"v\":\"%s\"}\n", m->facts[i].at, k, v);
    }
    return o < cap ? o : cap - 1;
}

void mlg_mind_memory_load(mlg_mind *m, const char *lines) {
    char line[1024], key[48], text[FACT_LEN], at[32];
    for (const char *p = lines; p && *p;) {
        const char *nl = strchr(p, '\n');
        const int n = nl ? (int)(nl - p) : (int)strlen(p);
        snprintf(line, sizeof line, "%.*s", n, p);
        field(line, "k", key, sizeof key), field(line, "v", text, sizeof text), field(line, "t", at, sizeof at);
        if (text[0]) add_fact(m, atoll(at), key, text);
        p = nl ? nl + 1 : NULL;
    }
}
const char *mlg_mind_memory_text(const mlg_mind *m, int i) { return i >= 0 && i < m->n_facts ? m->facts[i].text : ""; }

static const char *remember_prefixes[] = {
    "por favor ", "lembra que ", "lembre que ", "lembra de ", "lembre de ", "lembra ", "lembre ", "anota que ",
    "anota ", "guarda isso ", "guarda que ", "guarda ", "memoriza que ", "memoriza ", "nao esquece que ",
    "nao esqueca que ", "nao esquece de ", NULL};

/* "lembra que meu nome é Ana" keeps "meu nome é Ana"; returns the key it is about, if any */
static void remember(mlg_mind *m, const char *said, char *kept, int cap) {
    char f[1024], t[FACT_LEN];
    copy(t, sizeof t, said);
    trim(t);
    while (strlen(t) && strchr(".!", t[strlen(t) - 1])) t[strlen(t) - 1] = 0;
    for (int again = 1; again;) {  /* strip the request, keep the fact (in its original spelling) */
        again = 0;
        fold(t, f, sizeof f);
        for (const char **p = remember_prefixes; *p; p++)
            if (starts(f, *p) && strlen(t) > strlen(*p)) {
                memmove(t, t + strlen(*p), strlen(t) - strlen(*p) + 1);
                trim(t);
                if (t[0] == ':' || t[0] == ',') memmove(t, t + 1, strlen(t)), trim(t);
                again = 1;
                break;
            }
    }
    char key[48] = "", nf[FACT_LEN];
    norm(t, nf, sizeof nf);
    char *w[64];
    const int n = words(nf, w, 64);
    if (n >= 3 && (!strcmp(w[0], "meu") || !strcmp(w[0], "minha"))) {
        for (int i = 2; i < n && i < 6; i++)
            if (!strcmp(w[i], "e")) {  /* "meu X é", "minha cor favorita é" */
                key[0] = 0;
                for (int j = 1; j < i; j++) strncat(key, w[j], sizeof key - strlen(key) - 2), strcat(key, j + 1 < i ? " " : "");
                break;
            }
    } else if (n >= 3 && !strcmp(w[0], "eu") && !strcmp(w[1], "me") && !strcmp(w[2], "chamo")) {
        copy(key, sizeof key, "nome");
    }
    add_fact(m, m->now, key, t);
    if (m->memory_path[0]) {
        FILE *fh = fopen(m->memory_path, "w");  /* rewritten whole: replaced facts disappear */
        if (fh) {
            for (int i = 0; i < m->n_facts; i++) {
                fprintf(fh, "{\"t\":%lld,\"k\":", m->facts[i].at);
                json_str(fh, m->facts[i].key);
                fputs(",\"v\":", fh);
                json_str(fh, m->facts[i].text);
                fputs("}\n", fh);
            }
            fclose(fh);
        }
    }
    copy(kept, cap, t);
}

/* facts sharing the most words with the question; returns how many (best first) */
static int recall(const mlg_mind *m, const char *question, int *ids, int top) {
    char q[1024];
    norm(question, q, sizeof q);
    char *qw[64];
    const int nq = words(q, qw, 64);
    const int wants_name = has_word(qw, nq, "nome") || has_word(qw, nq, "chamo") ||
                           (has_word(qw, nq, "quem") && has_word(qw, nq, "sou"));
    int score[MAX_FACTS], n = 0;
    for (int i = 0; i < m->n_facts; i++) {
        char f[FACT_LEN];
        norm(m->facts[i].text, f, sizeof f);
        char *fw[64];
        const int nf = words(f, fw, 64);
        int s = 0;
        for (int a = 0; a < nq; a++)
            if (strlen(qw[a]) > 2 && strcmp(qw[a], "qual") && strcmp(qw[a], "que") && has_word(fw, nf, qw[a])) s++;
        if (wants_name && !strcmp(m->facts[i].key, "nome")) s += 3;
        score[i] = s;
    }
    for (int k = 0; k < top; k++) {
        int best = -1;
        for (int i = 0; i < m->n_facts; i++) {
            int used = 0;
            for (int j = 0; j < n; j++) used |= ids[j] == i;
            if (!used && score[i] > 0 && (best < 0 || score[i] > score[best] ||
                                          (score[i] == score[best] && m->facts[i].at >= m->facts[best].at)))
                best = i;
        }
        if (best < 0) break;
        ids[n++] = best;
    }
    return n;
}

/* "meu nome é Ana" -> "Seu nome é Ana." ; anything else is quoted back */
static void say_fact(const fact *f, char *out, int cap) {
    char low[FACT_LEN];
    fold(f->text, low, sizeof low);
    if (starts(low, "meu ")) snprintf(out, cap, "Seu %s.", f->text + 4);
    else if (starts(low, "minha ")) snprintf(out, cap, "Sua %s.", f->text + 6);
    else if (starts(low, "eu me chamo ")) snprintf(out, cap, "Seu nome é %s.", f->text + 12);
    else snprintf(out, cap, "Você me disse: \"%s\".", f->text);
}

/* ---- social ------------------------------------------------------------------- */
typedef struct {
    const char *match[12];  /* the normalized text must start with one of these */
    int whole;              /* ...and be nothing more than it (plus "mlg") */
    const char *reply[3];
} social_rule;

static const social_rule SOCIAL[] = {
    {{"oi", "oie", "oii", "oiii", "ola", "opa", "salve", "hey", "eae", "eai", "e ai", "fala"}, 1,
     {"Olá! Como posso ajudar?", "Oi! Estou aqui.", "E aí! Pode falar."}},
    {{"bom dia"}, 0, {"Bom dia! Estou funcionando e pronta para ajudar."}},
    {{"boa tarde"}, 0, {"Boa tarde! O que vamos fazer?"}},
    {{"boa noite"}, 0, {"Boa noite! Estou aqui."}},
    {{"tudo bem", "td bem", "tudo certo", "como vai", "como voce esta", "oi tudo bem", "ola tudo bem", "oi mlg tudo bem"}, 0,
     {"Tudo funcionando: meu corpo está estável e meus órgãos passaram no selo. E você?"}},
    {{"obrigado", "obrigada", "valeu", "vlw", "brigado", "brigada", "muito obrigado"}, 0,
     {"De nada!", "Disponha.", "Quando precisar."}},
    {{"tchau", "ate mais", "ate logo", "ate amanha", "falou", "flw", "tchau mlg"}, 0,
     {"Até mais! Vou consolidar minha memória enquanto isso."}},
    {{"otimo", "legal", "beleza", "blz", "show", "massa", "top", "certo", "ok", "certeza", "entendi", "perfeito"}, 1,
     {"Beleza.", "Combinado.", "Certo. Se quiser, me pergunte algo."}},
    {{"kk", "kkk", "kkkk", "haha", "hahaha", "rs", "rsrs"}, 1, {"Haha!"}},
};

static const char *social(mlg_mind *m, const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    int nw = n[0] ? 1 : 0;
    for (char *p = n; *p; p++) nw += *p == ' ';
    if (!n[0] || nw > 5) return NULL;
    /* "oi mlg" counts as "oi" */
    char *mlg = strstr(n, " mlg");
    if (mlg && mlg[4] == 0) *mlg = 0;
    for (size_t r = 0; r < sizeof SOCIAL / sizeof *SOCIAL; r++)
        for (int i = 0; i < 12 && SOCIAL[r].match[i]; i++) {
            const char *x = SOCIAL[r].match[i];
            const size_t lx = strlen(x);
            if (strncmp(n, x, lx) || (n[lx] && n[lx] != ' ') || (SOCIAL[r].whole && n[lx])) continue;
            int k = 0;
            while (k < 3 && SOCIAL[r].reply[k]) k++;
            return SOCIAL[r].reply[m->turn % k];
        }
    return NULL;
}

/* ---- self: what the MLG is and what this body can do now -------------------------- */
/* answered from the organs actually present, so it is true on every body and in every phase */
static const char *ABILITY[] = {"pode fazer", "sabe fazer", "consegue fazer", "voce faz", "vc faz", "tu faz", "voce serve",
                                "vc serve", "suas funcoes", "suas habilidades", "suas capacidades", "seus poderes",
                                "pode me ajudar", "consegue me ajudar", "da pra fazer", "posso fazer com voce",
                                "posso te perguntar", "posso perguntar", "como te uso", "como usar voce", NULL};
static const char *ABILITY_WHOLE[] = {"ajuda", "help", "comandos", "menu", "funcoes", "o que voce sabe", "oque voce sabe",
                                      "o que vc sabe", "oque vc sabe", "e agora", "o que eu faco", "oque eu faco", NULL};
static const char *IDENTITY_WHOLE[] = {"quem e voce", "quem e vc", "quem es tu", "quem e tu", "o que e voce", "oque e voce",
                                       "qual seu nome", "qual o seu nome", "qual e o seu nome", "qual e seu nome",
                                       "como voce se chama", "como vc se chama", "voce e quem", NULL};

static int asks_ability(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    char *mlg = strstr(n, " mlg");
    if (mlg && mlg[4] == 0) *mlg = 0;
    for (const char **x = ABILITY_WHOLE; *x; x++)
        if (!strcmp(n, *x)) return 1;
    int nw = n[0] ? 1 : 0;
    for (char *p = n; *p; p++) nw += *p == ' ';
    if (nw > 9 || strstr(n, "nao ")) return 0;  /* "o que você não sabe fazer" is the corpus's to answer */
    for (const char **x = ABILITY; *x; x++)
        if (strstr(n, *x)) return 1;
    return 0;
}

static int asks_identity(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    for (const char **x = IDENTITY_WHOLE; *x; x++)
        if (!strcmp(n, *x)) return 1;
    return 0;
}

static void abilities(const mlg_mind *m, char *out, int cap) {
    int o = snprintf(out, cap, "Agora eu sei: fazer contas exatas (\"15%% de 200\"), dizer a hora e a data, lembrar o que "
                               "você me contar (\"lembra que...\", \"meu nome é...\"), medir meu corpo (\"como está sua bateria?\")");
    if (m->know) {
        const int k = mlg_know_count(m->know);
        if (k >= 2000) o += snprintf(out + o, cap - o, ", responder pelo meu conhecimento (%d mil trechos)", k / 1000);
        else o += snprintf(out + o, cap - o, ", responder pelo meu conhecimento");
    }
    o += snprintf(out + o, cap - o, " e mostrar meu cérebro evoluindo (\"como está a evolução?\").");
    if (m->core) snprintf(out + o, cap - o, " Para o resto, quem conversa é meu Resident Core.");
    else snprintf(out + o, cap - o, " Meu Resident Core ainda está amadurecendo no laboratório; quando chegar, eu converso "
                                    "sobre qualquer coisa.");
}

/* nothing in the input says what is wanted: no word of three letters, no number */
static int says_nothing(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    char *w[64];
    const int nw = words(n, w, 64);
    for (int i = 0; i < nw; i++)
        if (has_digit(w[i]) || strlen(w[i]) >= 3) return 0;
    return 1;
}

/* the input, short enough to quote back */
static void quote(const char *text, char *out, int cap) {
    char q[64];
    int n = snprintf(q, sizeof q, "%s", text);
    if (n >= (int)sizeof q) {
        n = sizeof q - 1;
        while (n > 0 && ((unsigned char)q[n] & 0xC0) == 0x80) n--;  /* never cut a character in half */
        q[n] = 0;
        snprintf(out, cap, "%s…", q);
    } else {
        snprintf(out, cap, "%s", q);
    }
    trim(out);
}

static int first_person(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    char *w[64];
    const int nw = words(n, w, 64);
    static const char *ME[] = {"eu", "meu", "minha", "meus", "minhas", "mim", "comigo", NULL};
    for (int i = 0; i < nw; i++)
        for (const char **x = ME; *x; x++)
            if (!strcmp(w[i], *x)) return 1;
    return strstr(n, "me disse") || strstr(n, "te disse") || strstr(n, "te contei") || strstr(n, "me chamo");
}

/* a story, a poem, a joke, an opinion of its own: the Resident Core's to make, not a passage's to recite */
static int creative(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    static const char *X[] = {"conta uma historia", "conte uma historia", "contar uma historia", "inventa", "invente",
                              "cria um", "cria uma", "crie um", "crie uma", "escreve um", "escreva um", "escreve uma",
                              "escreva uma", "faz um poema", "faca um poema", "poema", "piada", "me conta algo",
                              "algo bonito", "sua cor favorita", "seu dia", "voce faria", "voce tem medo", "te deixa", NULL};
    for (const char **x = X; *x; x++)
        if (strstr(n, *x)) return 1;
    return 0;
}

static int asks_curiosity(const char *text) {
    char n[512];
    norm(text, n, sizeof n);
    static const char *X[] = {"curiosidade", "fato aleatorio", "fato interessante", "algo interessante", "me surpreend",
                              "voce sabia", "coisa interessante", "algo que eu nao sei", NULL};
    for (const char **x = X; *x; x++)
        if (strstr(n, *x)) return 1;
    return 0;
}

/* ---- procedural: arithmetic in words or symbols -------------------------------- */
static const struct { const char *w; int v; } NUMS[] = {
    {"zero", 0}, {"um", 1}, {"uma", 1}, {"dois", 2}, {"duas", 2}, {"tres", 3}, {"quatro", 4}, {"cinco", 5},
    {"seis", 6}, {"sete", 7}, {"oito", 8}, {"nove", 9}, {"dez", 10}, {"onze", 11}, {"doze", 12}, {"treze", 13},
    {"quatorze", 14}, {"catorze", 14}, {"quinze", 15}, {"dezesseis", 16}, {"dezessete", 17}, {"dezoito", 18},
    {"dezenove", 19}, {"vinte", 20}, {"trinta", 30}, {"quarenta", 40}, {"cinquenta", 50}, {"sessenta", 60},
    {"setenta", 70}, {"oitenta", 80}, {"noventa", 90}, {"cem", 100}, {"cento", 100}, {"duzentos", 200},
    {"trezentos", 300}, {"quatrocentos", 400}, {"quinhentos", 500}, {"seiscentos", 600}, {"setecentos", 700},
    {"oitocentos", 800}, {"novecentos", 900}};

static int num_word(const char *w) {
    for (size_t i = 0; i < sizeof NUMS / sizeof *NUMS; i++)
        if (!strcmp(NUMS[i].w, w)) return NUMS[i].v;
    return -1;
}

/* the question rewritten as symbols: "quanto é nove vezes nove" -> "quanto e 9 * 9" */
static void symbolize(const char *text, char *out, int cap) {
    char f[1024], tok[64];
    fold(text, f, sizeof f);
    for (char *p = f; *p; p++)  /* "3,5" is a decimal */
        if (*p == ',' && p > f && isdigit((unsigned char)p[-1]) && isdigit((unsigned char)p[1])) *p = '.';
    out[0] = 0;
    long long total = 0, acc = -1;
    int o = 0, i = 0;
#define EMIT(s) do { o += snprintf(out + o, o < cap ? cap - o : 0, "%s", s); } while (0)
    char numbuf[32];
    while (f[i]) {
        while (f[i] == ' ') i++;
        if (!f[i]) break;
        int t = 0;
        if (isalpha((unsigned char)f[i])) {
            while (isalpha((unsigned char)f[i]) && t < 63) tok[t++] = f[i++];
        } else {
            tok[t++] = f[i++];
            while (f[i] && (isdigit((unsigned char)f[i]) || f[i] == '.') && isdigit((unsigned char)tok[0]) && t < 63)
                tok[t++] = f[i++];
        }
        tok[t] = 0;
        if (!strcmp(tok, "por") && starts(f + i, " cento")) {  /* "por cento" is not "por 100" */
            if (acc >= 0 || total) {
                snprintf(numbuf, sizeof numbuf, " %lld ", total + (acc < 0 ? 0 : acc));
                EMIT(numbuf);
                total = 0, acc = -1;
            }
            EMIT(" por cento ");
            i += 6;
            continue;
        }
        const int v = num_word(tok);
        if (v >= 0) { acc = (acc < 0 ? 0 : acc) + v; continue; }
        if (!strcmp(tok, "mil")) { total += (acc < 1 ? 1 : acc) * 1000; acc = 0; continue; }
        if (!strcmp(tok, "e") && (acc >= 0 || total)) {  /* "vinte e cinco", unless "é" ends the number */
            int j = i;
            while (f[j] == ' ') j++;
            char nx[64];
            int k = 0;
            while (isalpha((unsigned char)f[j]) && k < 63) nx[k++] = f[j++];
            nx[k] = 0;
            if (num_word(nx) >= 0 || !strcmp(nx, "mil")) continue;
        }
        if (acc >= 0 || total) {
            snprintf(numbuf, sizeof numbuf, " %lld ", total + (acc < 0 ? 0 : acc));
            EMIT(numbuf);
            total = 0, acc = -1;
        }
        const char *sym = !strcmp(tok, "vezes") || !strcmp(tok, "x") ? "*" : !strcmp(tok, "mais") ? "+" :
                          !strcmp(tok, "menos") ? "-" : !strcmp(tok, "dividido") ? "/" :
                          !strcmp(tok, "multiplicado") ? "*" : NULL;
        if (sym) {
            EMIT(" ");
            EMIT(sym);
            EMIT(" ");
            if ((!strcmp(tok, "dividido") || !strcmp(tok, "multiplicado")) && starts(f + i, " por")) i += 4;
        } else {
            EMIT(tok);
            EMIT(" ");
        }
    }
    if (acc >= 0 || total) {
        snprintf(numbuf, sizeof numbuf, " %lld ", total + (acc < 0 ? 0 : acc));
        EMIT(numbuf);
    }
#undef EMIT
}

typedef struct {
    const char *s;
    int i, bad;
} expr;

static void sp(expr *e) { while (e->s[e->i] == ' ') e->i++; }
static double e_sum(expr *e);

static double e_atom(expr *e) {
    sp(e);
    if (e->s[e->i] == '(') {
        e->i++;
        const double v = e_sum(e);
        sp(e);
        if (e->s[e->i] == ')') e->i++;
        else e->bad = 1;
        return v;
    }
    if (e->s[e->i] == '-') { e->i++; return -e_atom(e); }
    char *end;
    const double v = strtod(e->s + e->i, &end);
    if (end == e->s + e->i) e->bad = 1;
    e->i = (int)(end - e->s);
    return v;
}

static double e_pow(expr *e) {
    const double b = e_atom(e);
    sp(e);
    if (e->s[e->i] == '^') { e->i++; return pow(b, e_pow(e)); }
    return b;
}

static double e_prod(expr *e) {
    double v = e_pow(e);
    for (;;) {
        sp(e);
        if (e->s[e->i] == '*') e->i++, v *= e_pow(e);
        else if (e->s[e->i] == '/') e->i++, v /= e_pow(e);
        else return v;
    }
}

static double e_sum(expr *e) {
    double v = e_prod(e);
    for (;;) {
        sp(e);
        if (e->s[e->i] == '+') e->i++, v += e_prod(e);
        else if (e->s[e->i] == '-') e->i++, v -= e_prod(e);
        else return v;
    }
}

/* Brazilian formatting: 1.234,5 */
static void fmt(double x, char *out, int cap) {
    if (isnan(x) || isinf(x)) { copy(out, cap, "indefinido"); return; }
    char raw[64];
    if (fabs(x - round(x)) < 1e-9 && fabs(x) < 1e15) snprintf(raw, sizeof raw, "%.0f", x);
    else {
        snprintf(raw, sizeof raw, "%.6f", x);
        int n = (int)strlen(raw);
        while (n && raw[n - 1] == '0') raw[--n] = 0;
        if (n && raw[n - 1] == '.') raw[--n] = 0;
    }
    char *dot = strchr(raw, '.');
    const int neg = raw[0] == '-', intlen = (int)(dot ? dot - raw : (long)strlen(raw)) - neg;
    int o = 0;
    if (neg) out[o++] = '-';
    for (int i = 0; i < intlen && o < cap - 2; i++) {
        out[o++] = raw[neg + i];
        if ((intlen - i - 1) % 3 == 0 && i < intlen - 1) out[o++] = '.';
    }
    if (dot) {
        out[o++] = ',';
        for (const char *p = dot + 1; *p && o < cap - 1; p++) out[o++] = *p;
    }
    out[o] = 0;
}

static int procedural_math(const char *text, char *out, int cap) {
    char s[1024], a[64], b[64];
    symbolize(text, s, sizeof s);
    double x, y;
    const char *p;
    if ((p = strstr(s, "raiz")) && sscanf(p, "raiz quadrada de %lf", &x) == 1) {
        fmt(x, a, sizeof a), fmt(sqrt(x), b, sizeof b);
        return snprintf(out, cap, "A raiz quadrada de %s é %s.", a, b);
    }
    if ((p = strstr(s, "raiz")) && sscanf(p, "raiz de %lf", &x) == 1) {
        fmt(x, a, sizeof a), fmt(sqrt(x), b, sizeof b);
        return snprintf(out, cap, "A raiz quadrada de %s é %s.", a, b);
    }
    for (p = s; *p; p++) {
        if (!isdigit((unsigned char)*p) && *p != '-') continue;
        char unit[64] = "";
        if (sscanf(p, "%lf ao %63s", &x, unit) == 2 && (!strcmp(unit, "quadrado") || !strcmp(unit, "cubo"))) {
            fmt(x, a, sizeof a), fmt(!strcmp(unit, "cubo") ? x * x * x : x * x, b, sizeof b);
            return snprintf(out, cap, "%s ao %s é %s.", a, unit, b);
        }
        if (sscanf(p, "%lf por cento de %lf", &x, &y) == 2 || sscanf(p, "%lf %% de %lf", &x, &y) == 2) {
            char c[64];
            fmt(x, a, sizeof a), fmt(y, b, sizeof b), fmt(x * y / 100, c, sizeof c);
            return snprintf(out, cap, "%s%% de %s é %s.", a, b, c);
        }
        if (sscanf(p, "%lf km em %63s", &x, unit) == 2 && starts(unit, "metro")) {
            fmt(x, a, sizeof a), fmt(x * 1000, b, sizeof b);
            return snprintf(out, cap, "%s km são %s metros.", a, b);
        }
        if (strstr(p, "fahrenheit") && sscanf(p, "%lf graus", &x) == 1) {
            fmt(x, a, sizeof a), fmt(x * 9 / 5 + 32, b, sizeof b);
            return snprintf(out, cap, "%s °C são %s °F.", a, b);
        }
        break;
    }
    if ((p = strstr(s, "minutos")) && strstr(s, "hora")) {
        for (p = s; *p && !isdigit((unsigned char)*p); p++) {}
        if (*p && sscanf(p, "%lf", &x) == 1) {
            fmt(x, a, sizeof a), fmt(x * 60, b, sizeof b);
            return snprintf(out, cap, "%s horas têm %s minutos.", a, b);
        }
    }
    /* the longest run of digits and operators, if it holds an operator */
    int best_at = -1, best_len = 0;
    for (int i = 0; s[i];) {
        if (strchr("0123456789.()-", s[i]) && (i == 0 || !isalpha((unsigned char)s[i - 1]))) {
            int j = i, ops = 0, digits = 0;
            while (s[j] && strchr("0123456789.+-*/^() ", s[j])) {
                ops += !!strchr("+-*/^", s[j]) && j > i;
                digits += !!isdigit((unsigned char)s[j]);
                j++;
            }
            if (ops && digits && j - i > best_len) best_at = i, best_len = j - i;
            i = j > i ? j : i + 1;
        } else {
            i++;
        }
    }
    if (best_at < 0) return 0;
    char ex[256];
    snprintf(ex, sizeof ex, "%.*s", best_len, s + best_at);
    trim(ex);
    while (strlen(ex) && strchr("+-*/^", ex[strlen(ex) - 1])) ex[strlen(ex) - 1] = 0, trim(ex);
    expr e = {ex, 0, 0};
    const double v = e_sum(&e);
    sp(&e);
    if (e.bad || ex[e.i]) return 0;
    char pretty[512];
    int o = 0;
    for (const char *q = ex; *q && o < (int)sizeof pretty - 4; q++) {
        if (*q == ' ' && (o == 0 || pretty[o - 1] == ' ')) continue;
        if (*q == '*') o += snprintf(pretty + o, sizeof pretty - o, "×");
        else if (*q == '/') o += snprintf(pretty + o, sizeof pretty - o, "÷");
        else if (*q == '.' ) pretty[o++] = ',';
        else pretty[o++] = *q;
    }
    pretty[o] = 0;
    fmt(v, a, sizeof a);
    return snprintf(out, cap, "%s = %s", pretty, a);
}

/* ---- procedural: time ---------------------------------------------------------- */
static const char *WEEKDAY[] = {"domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
                                "sexta-feira", "sábado"};
static const char *MONTH[] = {"janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
                              "setembro", "outubro", "novembro", "dezembro"};

static void procedural_time(const mlg_mind *m, const char *question, char *out, int cap) {
    const time_t t = (time_t)(m->now + m->tz_minutes * 60LL);
    struct tm tm;
    gmtime_r(&t, &tm);
    char q[512];
    norm(question, q, sizeof q);
    char *w[64];
    const int n = words(q, w, 64);
    if (has_word(w, n, "hora") || has_word(w, n, "horas"))
        snprintf(out, cap, "Agora são %02d:%02d.", tm.tm_hour, tm.tm_min);
    else if (has_word(w, n, "semana"))
        snprintf(out, cap, "Hoje é %s.", WEEKDAY[tm.tm_wday]);
    else if (has_word(w, n, "mes"))
        snprintf(out, cap, "Estamos em %s de %d.", MONTH[tm.tm_mon], tm.tm_year + 1900);
    else if (has_word(w, n, "ano"))
        snprintf(out, cap, "Estamos em %d.", tm.tm_year + 1900);
    else
        snprintf(out, cap, "Hoje é %s, %d de %s de %d, %02d:%02d.", WEEKDAY[tm.tm_wday], tm.tm_mday,
                 MONTH[tm.tm_mon], tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
}

/* ---- body and lab: facts the host reports ---------------------------------------- */
static int body_get(const mlg_mind *m, const char *key, char *out, int cap) {
    char pat[64];
    snprintf(pat, sizeof pat, "%s=", key);
    for (const char *p = m->body; (p = strstr(p, pat)); p++)
        if (p == m->body || p[-1] == '\n') {
            p += strlen(pat);
            int o = 0;
            while (*p && *p != '\n' && o < cap - 1) out[o++] = *p++;
            out[o] = 0;
            return 1;
        }
    out[0] = 0;
    return 0;
}

static void body_report(const mlg_mind *m, int asks_heat, char *out, int cap) {
    char dev[128], cores[16], bat[16], temp[16], bps[16], threads[16], thermal[16];
    int o = 0;
#define ADD(...) do { if (o < cap) o += snprintf(out + o, cap - o, __VA_ARGS__); } while (0)
    if (body_get(m, "device", dev, sizeof dev)) ADD("Meu corpo é um %s", dev);
    else ADD("Ainda não medi este corpo");
    if (body_get(m, "cores", cores, sizeof cores)) ADD(" com %s núcleos", cores);
    ADD(". ");
    const int has_bat = body_get(m, "battery", bat, sizeof bat), has_temp = body_get(m, "temperature", temp, sizeof temp);
    if (has_bat && has_temp) ADD("Bateria em %s%%, %s °C. ", bat, temp);
    else if (has_bat) ADD("Bateria em %s%%. ", bat);
    else if (has_temp) ADD("Estou a %s °C. ", temp);
    if (body_get(m, "bytes_per_s", bps, sizeof bps) && body_get(m, "threads", threads, sizeof threads))
        ADD("Penso a %s bytes por segundo com %s %s, medido aqui. ", bps, threads, atoi(threads) == 1 ? "thread" : "threads");
    if (body_get(m, "thermal", thermal, sizeof thermal) && atoi(thermal) >= 3) ADD("Estou quente: vou pensar com menos threads. ");
    if (asks_heat) ADD("Quando a temperatura sobe, reduzo o nível de raciocínio e uso menos threads até o processador esfriar.");
#undef ADD
    trim(out);
}

/* Share of the question's content words found in a passage: BM25 rewards one
 * rare word, but evidence has to be about what was asked. */
static const char *STOP[] = {"quem", "qual", "quais", "como", "onde", "quando", "voce", "sobre", "isso", "essa", "esse",
                             "para", "pelo", "pela", "entre", "mais", "muito", "uma", "umas", "uns", "que", "por",
                             "dos", "das", "nos", "nas", "com", "tem", "sao", "foi", "era", "ser", "fale", "fala",
                             "conta", "explica", "sabe", "diga", "diz", "quanto", "quanta", "quantos", "quantas",
                             /* the frame of a question, not what it asks about */
                             "funciona", "funcionam", "significa", "serve", "servem", "existe", "existem", "acontece",
                             "aconteceu", "fale", "falar", "explique", "explicar", "conte", "contar", "defina", "definir",
                             "descreva", "descrever", "resuma", "resumir", "gosta", "acha", "pensa", NULL};

static int stopword(const char *w) {
    for (const char **s = STOP; *s; s++)
        if (!strcmp(*s, w)) return 1;
    return 0;
}

static int has_digit(const char *w) {
    for (; *w; w++)
        if (isdigit((unsigned char)*w)) return 1;
    return 0;
}

/* a word that says what is asked: not a stop word, and long enough unless it is a number */
static int content_word(const char *w) { return has_digit(w) || (strlen(w) >= 3 && !stopword(w)); }

/* Dice over content words; *n gets how many the question has. Numbers must
 * all match: "15 por cento de 200" is not "20 por cento de 76". */
static double content_dice(const char *a, const char *b, int *n) {
    char x[512], y[512];
    norm(a, x, sizeof x), norm(b, y, sizeof y);
    char *wa[64], *wb[64];
    const int na = words(x, wa, 64), nb = words(y, wb, 64);
    int ca = 0, cb = 0, common = 0;
    for (int i = 0; i < na; i++)
        if (content_word(wa[i])) {
            ca++;
            int hit = 0;
            for (int j = 0; j < nb; j++) hit |= !strcmp(wa[i], wb[j]);
            if (!hit && has_digit(wa[i])) { *n = ca; return 0.0; }
            common += hit;
        }
    for (int j = 0; j < nb; j++) {
        if (!content_word(wb[j])) continue;
        cb++;
        int hit = 0;
        for (int i = 0; i < na; i++) hit |= !strcmp(wa[i], wb[j]);
        if (!hit && has_digit(wb[j])) { *n = ca; return 0.0; }
    }
    *n = ca;
    return ca + cb ? 2.0 * common / (ca + cb) : 0.0;
}

/* How much a word of the question tells, from the index: rare words weigh, common
 * ones do not. A word the index does not hold was too common to keep if the
 * passage has it, and is unheard of otherwise. */
static float weight(const mlg_mind *m, const char *w, int in_passage) {
    if (has_digit(w)) return 4.f;
    if (strlen(w) < 3 || stopword(w)) return 0.f;
    if (!m->know) return 1.f;
    const float idf = mlg_know_idf(m->know, w, (int)strlen(w));
    return idf >= 0.f ? idf : in_passage ? 0.f : 8.f;
}

/* the word, or a shared stem of 6 ("descobriu" finds "descobrir") */
static int has_stem(char **w, int n, const char *x) {
    if (has_word(w, n, x)) return 1;
    for (int j = 0; strlen(x) >= 6 && j < n; j++)
        if (strlen(w[j]) >= 6 && !strncmp(x, w[j], 6)) return 1;
    return 0;
}

/* share of what the question asks (weighted by rarity) that a passage contains */
static double coverage(const mlg_mind *m, const char *question, const char *passage) {
    char q[512], pn[1600];
    norm(question, q, sizeof q);
    norm(passage, pn, sizeof pn);
    char *qw[64], *pw[300];
    const int nq = words(q, qw, 64), np = words(pn, pw, 300);
    double asked = 0.0, found = 0.0;
    for (int i = 0; i < nq; i++) {
        const int hit = has_stem(pw, np, qw[i]);
        const float wt = weight(m, qw[i], hit);
        asked += wt, found += hit ? wt : 0.f;
    }
    return asked > 0.0 ? found / asked : 0.0;
}

/* ---- the question my own corpus answers word for word --------------------------- */
/* passages of the MLG corpus are titled "§ q1 | q2 | ..." (mlgsub/know.py); the
 * mark is not a word, so it does not weigh in retrieval */
static const char MLG_TITLE[] = "\xc2\xa7 ";

static double overlap(const char *a, const char *b) {
    char x[512], y[512];
    norm(a, x, sizeof x), norm(b, y, sizeof y);
    char *wa[64], *wb[64];
    const int na = words(x, wa, 64), nb = words(y, wb, 64);
    int common = 0, la = 0, lb = 0;
    for (int i = 0; i < na; i++) la += strlen(wa[i]) > 1;
    for (int i = 0; i < nb; i++) lb += strlen(wb[i]) > 1;
    for (int i = 0; i < na; i++)
        if (strlen(wa[i]) > 1 && has_word(wb, nb, wa[i])) common++;
    return la + lb ? 2.0 * common / (la + lb) : 0.0;
}

static int own_answer(const mlg_mind *m, const char *question, int id, char *out, int cap) {
    int tl, xl;
    const char *title = mlg_know_title(m->know, id, &tl), *text = mlg_know_text(m->know, id, &xl);
    if (tl <= (int)strlen(MLG_TITLE) || strncmp(title, MLG_TITLE, strlen(MLG_TITLE))) return 0;
    char qs[1024];
    snprintf(qs, sizeof qs, "%.*s", tl - (int)strlen(MLG_TITLE), title + strlen(MLG_TITLE));
    for (char *q = qs, *bar; q && *q; q = bar ? bar + 3 : NULL) {
        bar = strstr(q, " | ");
        if (bar) *bar = 0;
        int content;
        const double c = content_dice(question, q, &content);
        if (content ? c >= 0.8 : overlap(question, q) >= 0.9) {
            snprintf(out, cap, "%.*s", xl, text);
            /* the corpus remembers the body the MLG was first designed for; the body it is in now is measured */
            char dev[128];
            if (strstr(out, "Raspberry") && body_get(m, "device", dev, sizeof dev) && !strstr(dev, "Raspberry"))
                snprintf(out + strlen(out), cap - strlen(out), " Hoje meu corpo é um %s, medido por mim.", dev);
            return 1;
        }
    }
    return 0;
}

/* x cut to fit cap bytes: at a sentence end when one falls in the second half, never inside a character */
static int clip(const char *x, int xl, int cap) {
    if (xl < cap) return xl;
    int n = cap - 1;
    while (n > 0 && ((unsigned char)x[n] & 0xC0) == 0x80) n--;
    for (int i = n - 1; i > n / 2; i--)
        if (x[i] == '.' && x[i + 1] == ' ') return i + 1;
    return n;
}

/* a Portuguese plural made singular, as titles are written ("buracos negros" -> "buraco negro") */
static void singular(const char *w, char *out, int cap) {
    const int n = (int)strlen(w);
    if (n > 4 && !strcmp(w + n - 3, "oes")) snprintf(out, cap, "%.*sao", n - 3, w);
    else if (n > 4 && !strcmp(w + n - 3, "aes")) snprintf(out, cap, "%.*sao", n - 3, w);
    else if (n > 4 && (!strcmp(w + n - 3, "res") || !strcmp(w + n - 3, "zes") || !strcmp(w + n - 3, "ses")))
        snprintf(out, cap, "%.*s", n - 2, w);
    else if (n > 3 && w[n - 1] == 's') snprintf(out, cap, "%.*s", n - 1, w);
    else snprintf(out, cap, "%s", w);
}

/* the run w[i..i+len) as a title key, its words as written or made singular */
static int run_key(char **w, int i, int len, int sing, char *out, int cap) {
    int o = 0;
    for (int j = i; j < i + len && o < cap - 1; j++) {
        char x[128];
        if (sing) singular(w[j], x, sizeof x);
        else snprintf(x, sizeof x, "%s", w[j]);
        o += snprintf(out + o, cap - o, "%s%s", j > i ? " " : "", x);
    }
    return o < cap ? o : -1;
}

/* The article the question names: the run of its words, rarest overall, that is
 * a title in the index ("quem foi pedro alvares cabral", "me fale de portugal"),
 * or else the title of a passage found that holds every word asked about
 * ("napoleao" -> "Napoleão Bonaparte"). It must carry a good part of what is
 * asked, or it is a word in passing. */
static int subject(const mlg_mind *m, const char *text, const int *hits, int n_hits) {
    char n[512];
    norm(text, n, sizeof n);
    char *w[64];
    const int nw = words(n, w, 64);
    double asked = 0.0, best_w = 0.0;
    int best = -1;
    for (int i = 0; i < nw; i++) asked += weight(m, w[i], 0);
    for (int i = 0; i < nw; i++)
        for (int len = 1; len <= 6 && i + len <= nw; len++) {
            if (weight(m, w[i], 0) == 0.f || weight(m, w[i + len - 1], 0) == 0.f) continue;
            double wt = 0.0;
            for (int j = i; j < i + len; j++) wt += weight(m, w[j], 1);
            if (wt <= best_w) continue;
            for (int sing = 0; sing < 2; sing++) {
                char g[256];
                const int o = run_key(w, i, len, sing, g, sizeof g);
                const int id = o > 0 ? mlg_know_lookup(m->know, g, o) : -1;
                if (id >= 0) { best_w = wt, best = id; break; }
            }
        }
    if (best < 0)
        for (int h = 0; h < n_hits && h < 5; h++) {
            int tl;
            const char *title = mlg_know_title(m->know, hits[h], &tl);
            if (tl >= 2 && !strncmp(title, MLG_TITLE, strlen(MLG_TITLE))) continue;
            char tn[512];
            const int k = mlg_normalize(title, tl, tn, sizeof tn - 1);
            tn[k] = 0;
            char *tw[32];
            const int ntw = words(tn, tw, 32);
            int all = 1, any = 0;
            for (int i = 0; i < nw; i++)
                if (weight(m, w[i], 0) > 0.f) any = 1, all &= has_word(tw, ntw, w[i]);
            if (any && all) { best = mlg_know_lead(m->know, hits[h]), best_w = asked; break; }
        }
    return asked > 0.0 && best_w >= 0.4 * asked ? best : -1;
}

/* ---- thinking ------------------------------------------------------------------ */
static const char *fallback_intent(const char *text) {
    char n[512], f[512];
    norm(text, n, sizeof n);
    fold(text, f, sizeof f);
    char *w[64];
    char nb[512];
    copy(nb, sizeof nb, n);
    const int nw = words(nb, w, 64);
    char probe[512];
    if (procedural_math(text, probe, sizeof probe) > 0) return "math";
    if (strstr(n, "que horas") || strstr(n, "que dia") || strstr(n, "data de hoje")) return "time";
    if (starts(n, "lembr") || starts(n, "anota") || starts(n, "guarda") || starts(n, "memoriza") ||
        starts(n, "meu nome e") || starts(n, "eu me chamo"))
        return "remember";
    if ((strstr(n, "meu nome") || strstr(n, "me chamo") || strstr(n, "o que eu te disse") || strstr(n, "o que eu disse")) &&
        (has_word(w, nw, "qual") || has_word(w, nw, "como") || has_word(w, nw, "lembra") || has_word(w, nw, "sabe")))
        return "recall";
    if (strstr(n, "bateria") || strstr(n, "temperatura") || strstr(n, "seu corpo")) return "body";
    if (strstr(n, "evolu") || strstr(n, "cerebro") || strstr(n, "laborat") || strstr(n, "colab")) return "lab";
    if (nw <= 3) return "chat";
    return "knowledge";
}

static void json_escape(const char *s, char *out, int cap) {
    int o = 0;
    for (; *s && o < cap - 7; s++) {
        if (*s == '"' || *s == '\\') out[o++] = '\\', out[o++] = *s;
        else if (*s == '\n') out[o++] = '\\', out[o++] = 'n';
        else if ((unsigned char)*s < 0x20) o += snprintf(out + o, cap - o, "\\u%04x", *s);
        else out[o++] = *s;
    }
    out[o] = 0;
}

int mlg_mind_think(mlg_mind *m, const char *text, mlg_thought *t) {
    memset(t, 0, sizeof *t);
    m->turn++;
    copy(t->organ, sizeof t->organ, "core");
    /* the router's verdict */
    if (m->router) {
        float p[256];
        const int best = mlg_router_route(m->router, text, p);
        t->n_intents = mlg_router_count(m->router) < 16 ? mlg_router_count(m->router) : 16;
        for (int i = 0; i < t->n_intents; i++) t->probs[i] = p[i];
        copy(t->intent, sizeof t->intent, mlg_router_name(m->router, best));
    } else {
        copy(t->intent, sizeof t->intent, fallback_intent(text));
    }
    const char *in = t->intent;
    char buf[2048];
    const char *s;
    /* greetings may be routed as chat or as a question: the social expert answers both */
    if ((!strcmp(in, "chat") || !strcmp(in, "knowledge")) && (s = social(m, text))) {
        copy(t->organ, sizeof t->organ, "social"), copy(t->reply, sizeof t->reply, s);
        return t->final = 1;
    }
    /* recalling is about the person ("meu nome", "o que eu disse"), not about the MLG ("sua cor favorita") */
    if (!strcmp(in, "recall") && !first_person(text)) copy(t->intent, sizeof t->intent, "chat");
    const int about_user = !strcmp(in, "remember") || !strcmp(in, "recall");
    if (!about_user && asks_ability(text)) {
        abilities(m, t->reply, sizeof t->reply);
        copy(t->organ, sizeof t->organ, "self");
        return t->final = 1;
    }
    /* a curiosity: the opening of an article picked at random, not words a brain made up */
    if (!about_user && m->know && asks_curiosity(text)) {
        uint64_t r = (uint64_t)m->now * 6364136223846793005ULL + m->turn * 1442695040888963407ULL + 1;
        const int nd = mlg_know_count(m->know);
        for (int tries = 0; tries < 200 && nd > 0; tries++) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            const int id = mlg_know_lead(m->know, (int)((r >> 33) % (uint64_t)nd));
            int tl, xl;
            const char *title = mlg_know_title(m->know, id, &tl), *x = mlg_know_text(m->know, id, &xl);
            if (tl < 2 || tl > 80 || !strncmp(title, MLG_TITLE, strlen(MLG_TITLE)) || xl < 200) continue;
            int end = 0;  /* the first sentence, if it is a whole one of a reasonable size */
            for (int i = 60; i < xl && i < 320 && !end; i++)
                if (x[i] == '.' && (i + 1 == xl || x[i + 1] == ' ') && !(x[i - 2] == ' ' || x[i - 2] == '.')) end = i + 1;
            if (!end) continue;
            snprintf(t->evidence_title, sizeof t->evidence_title, "%.*s", tl, title);
            snprintf(t->evidence, sizeof t->evidence, "%.*s", clip(x, xl, sizeof t->evidence), x);
            snprintf(t->reply, sizeof t->reply, "Uma curiosidade, sobre %.*s: %.*s", tl, title, end, x);
            copy(t->organ, sizeof t->organ, "knowledge");
            return t->final = 1;
        }
    }
    if (says_nothing(text)) {
        char q[96];
        quote(text, q, sizeof q);
        static const char *ASK[] = {"Não entendi \"%s\". Pode falar com mais palavras? Para saber o que eu faço, pergunte "
                                    "\"o que você sabe fazer?\".",
                                    "\"%s\"? Me diga um pouco mais.", "Não peguei o que você quis dizer com \"%s\". Pode repetir?"};
        snprintf(t->reply, sizeof t->reply, ASK[m->turn % 3], q);
        copy(t->organ, sizeof t->organ, "self");
        return t->final = 1;
    }
    /* an expression the procedural expert can compute exactly, whatever the router thought */
    if (!about_user && procedural_math(text, buf, sizeof buf) > 0) {
        copy(t->organ, sizeof t->organ, "procedural"), copy(t->reply, sizeof t->reply, buf);
        return t->final = 1;
    }
    /* a question the MLG's own corpus answers, word for word */
    if (!about_user && strcmp(in, "time") && m->know) {
        /* asked exactly as the corpus asks it, then as nearly as retrieval finds */
        int ids[17];
        float sc[17];
        char nq[512];
        norm(text, nq, sizeof nq);
        char *tail = strstr(nq, " mlg");
        if (tail && tail[4] == 0 && tail != nq) *tail = 0;  /* "o que você sabe, mlg" asks the same */
        ids[0] = mlg_know_asked(m->know, nq, (int)strlen(nq)), sc[0] = 0.f;
        const int n = 1 + mlg_know_search(m->know, text, 16, ids + 1, sc + 1);
        for (int i = ids[0] < 0; i < n; i++)
            if (own_answer(m, text, ids[i], buf, sizeof buf)) {
                int tl;
                const char *title = mlg_know_title(m->know, ids[i], &tl);
                snprintf(t->evidence_title, sizeof t->evidence_title, "%.*s", tl, title);
                copy(t->evidence, sizeof t->evidence, buf);
                t->evidence_score = sc[i];
                copy(t->organ, sizeof t->organ, "knowledge"), copy(t->reply, sizeof t->reply, buf);
                return t->final = 1;
            }
    }
    /* who it is, on a body whose corpus has not arrived */
    if (asks_identity(text)) {
        copy(t->reply, sizeof t->reply, "Sou a MLG: uma mente que nasceu do zero, sem modelo pronto. Meço o corpo em que estou, "
                                        "aprendo no laboratório e cada cérebro meu só chega aqui depois de passar no selo.");
        copy(t->organ, sizeof t->organ, "self");
        return t->final = 1;
    }
    if (!strcmp(in, "time")) {
        procedural_time(m, text, t->reply, sizeof t->reply);
        copy(t->organ, sizeof t->organ, "procedural");
        return t->final = 1;
    }
    if (!strcmp(in, "remember")) {
        remember(m, text, buf, sizeof buf);
        snprintf(t->reply, sizeof t->reply, "Guardei na minha memória: \"%s\".", buf);
        copy(t->organ, sizeof t->organ, "memory");
        return t->final = 1;
    }
    if (!strcmp(in, "recall")) {
        int ids[3];
        const int n = recall(m, text, ids, 2);
        copy(t->organ, sizeof t->organ, "memory");
        if (!n) copy(t->reply, sizeof t->reply, "Ainda não tenho isso guardado. Me diga e eu lembro.");
        for (int i = 0, o = 0; i < n; i++) {
            say_fact(&m->facts[ids[i]], buf, sizeof buf);
            o += snprintf(t->reply + o, sizeof t->reply - o, "%s%s", i ? " " : "", buf);
        }
        return t->final = 1;
    }
    if (!strcmp(in, "body")) {
        char q[512];
        norm(text, q, sizeof q);
        body_report(m, strstr(q, "temperatura") || strstr(q, "quente") || strstr(q, "esquent") || strstr(q, "calor") != NULL,
                    t->reply, sizeof t->reply);
        copy(t->organ, sizeof t->organ, "body");
        return t->final = 1;
    }
    if (!strcmp(in, "lab")) {
        copy(t->reply, sizeof t->reply, m->lab[0] ? m->lab :
             "Estou sem ligação com o laboratório agora. Quando o Colab rodar, eu vejo a evolução ao vivo.");
        copy(t->organ, sizeof t->organ, "lab");
        return t->final = 1;
    }
    /* knowledge, chat, or arithmetic it could not do exactly: evidence first */
    char ctx[1400] = "";
    if (m->know && !creative(text)) {
        /* searched by what it asks about: "quantos" and "tem" would find every "quantos ... tem" */
        char nq[512], query[512] = "";
        norm(text, nq, sizeof nq);
        char *qw[64];
        const int nqw = words(nq, qw, 64);
        for (int i = 0, o = 0; i < nqw; i++)
            if (weight(m, qw[i], 0) > 0.f) o += snprintf(query + o, sizeof query - o, "%s%s", o ? " " : "", qw[i]);
        int ids[16];
        float sc[16];
        const int n = query[0] ? mlg_know_search(m->know, query, 16, ids, sc) : 0;
        /* the article the question names speaks first: the passage of it that covers the
         * question best ("população de Angola"), its lead otherwise */
        const int lead = subject(m, text, ids, n);
        int first = lead;
        if (lead >= 0) {
            double best_c = -1.0;
            for (int i = -1; i < n; i++) {
                const int id = i < 0 ? lead : ids[i];
                if (i >= 0 && (id == lead || mlg_know_lead(m->know, id) != lead)) continue;
                int xl, tl;
                const char *x = mlg_know_text(m->know, id, &xl), *title = mlg_know_title(m->know, id, &tl);
                char whole[1600];
                snprintf(whole, sizeof whole, "%.*s %.*s", tl, title, xl < 1200 ? xl : 1200, x);
                const double c = coverage(m, text, whole);
                if (c > best_c + 1e-6) best_c = c, first = id;
            }
        }
        int used = 0;
        for (int i = -2; i < n && used < 2; i++) {
            const int id = i == -2 ? first : i == -1 ? (lead != first ? lead : -1) : ids[i];
            if (id < 0 || (i >= 0 && (id == lead || id == first))) continue;
            if (i >= 0 && (sc[i] < 4.f || (used && sc[i] < sc[0] * 0.6f))) break;  /* weak matches are not evidence */
            int xl, tl;
            const char *x = mlg_know_text(m->know, id, &xl), *title = mlg_know_title(m->know, id, &tl);
            char whole[1600];
            snprintf(whole, sizeof whole, "%.*s %.*s", tl, title, xl < 1200 ? xl : 1200, x);
            if (i >= 0 && coverage(m, text, whole) < 0.6) continue;  /* about something else */
            if (!used++) {
                snprintf(t->evidence_title, sizeof t->evidence_title, "%.*s", tl, title);
                snprintf(t->evidence, sizeof t->evidence, "%.*s", clip(x, xl, sizeof t->evidence), x);
                t->evidence_score = i < 0 ? 0.f : sc[i];
            }
            int keep = xl < 420 ? xl : 420;  /* the cut the lab trained with (mlgsub/know.py CONTEXT) */
            while (keep < xl && keep > 0 && (x[keep] & 0xC0) == 0x80) keep--;
            if (keep < xl) while (keep > 0 && x[keep] != ' ') keep--;
            snprintf(ctx + strlen(ctx), sizeof ctx - strlen(ctx), "%s%.*s", ctx[0] ? " " : "", keep, x);
        }
    }
    int ids[2];
    const int nf = recall(m, text, ids, 2);
    for (int i = 0; i < nf; i++)
        snprintf(ctx + strlen(ctx), sizeof ctx - strlen(ctx), "%s(memória) %s", ctx[0] ? " " : "", m->facts[ids[i]].text);
    char q[1024];
    copy(q, sizeof q, text);
    trim(q);
    snprintf(t->prompt, sizeof t->prompt, "%s%s%sU: %s\nM:", ctx[0] ? "C: " : "", ctx, ctx[0] ? "\n" : "", q);
    if (t->evidence[0]) copy(t->organ, sizeof t->organ, "core");
    return t->final = 0;
}

/* The sentences of the evidence that answer the question best (by what they share
 * with it, rare words weighing most), in their order. */
static void extract(const mlg_mind *m, const char *question, const char *evidence, char *out, int cap) {
    char ev[1600], q[512];
    copy(ev, sizeof ev, evidence);
    norm(question, q, sizeof q);
    char *qw[64];
    const int nq = words(q, qw, 64);
    char *sent[48];
    int ns = 0;
    char *st = ev;
    for (char *p = ev; *p && ns < 48; p++)  /* sentences end at . ! ? followed by a space */
        if (strchr(".!?", *p) && p[1] == ' ') {
            p[1] = 0;
            sent[ns++] = st;
            st = p + 2;
            p++;
        }
    if (*st && ns < 48) sent[ns++] = st;
    if (!ns) { copy(out, cap, evidence); return; }
    int best = 0;
    double best_s = -1.0;
    for (int i = 0; i < ns; i++) {
        char sn[1600];
        norm(sent[i], sn, sizeof sn);
        char *sw[96];
        const int nw = words(sn, sw, 96);
        double sc = 0.0;
        for (int a = 0; a < nq; a++)
            if (has_stem(sw, nw, qw[a])) sc += weight(m, qw[a], 1);
        if (sc > best_s + 1e-6) best_s = sc, best = i;
    }
    if (best + 1 < ns && strlen(sent[best]) < 160) snprintf(out, cap, "%s %s", sent[best], sent[best + 1]);
    else copy(out, cap, sent[best]);
}

/* share of the answer's content words that the evidence or the question contain */
static double grounded(const char *answer, const char *evidence, const char *question) {
    char a[2048], e[2600], en[2600];
    norm(answer, a, sizeof a);
    snprintf(e, sizeof e, "%s %s", evidence, question);
    norm(e, en, sizeof en);
    char *aw[160], *ew[480];
    const int na = words(a, aw, 160), ne = words(en, ew, 480);
    int content = 0, found = 0;
    for (int i = 0; i < na; i++)
        if (strlen(aw[i]) > 3) content++, found += has_word(ew, ne, aw[i]);
    return content ? (double)found / content : 1.0;
}

/* a byte-level brain gone astray: the same word, or the same pair, over and over */
static int degenerate(const char *answer) {
    char a[2048];
    norm(answer, a, sizeof a);
    char *w[300];
    const int n = words(a, w, 300);
    int run = 1;
    for (int i = 1; i < n; i++) {
        run = !strcmp(w[i], w[i - 1]) ? run + 1 : 1;
        if (run >= 3) return 1;
    }
    for (int i = 0; i + 3 < n; i++) {
        int rep = 0;
        for (int j = i + 2; j + 1 < n; j++)
            if (!strcmp(w[i], w[j]) && !strcmp(w[i + 1], w[j + 1]) && strlen(w[i]) > 2 && ++rep >= 3) return 1;
    }
    return 0;
}

/* A byte-level brain that has not learned words yet: runs of punctuation, or words
 * that no passage the MLG knows has ever used ("iénos-sístolôntios,,,mõurcos"). */
static int gibberish(const mlg_mind *m, const char *answer) {
    int run = 1, punct = 0, chars = 0;
    for (const char *p = answer; *p; p++) {
        if (p > answer && *p == p[-1] && strchr(",.;:!?-_*'\"()", *p) && ++run >= 4) return 1;
        if (p == answer || *p != p[-1]) run = 1;
        chars++;
        punct += strchr(",.;:!?-_*'\"()[]{}/\\", *p) != NULL;
    }
    if (chars > 20 && punct * 5 > chars) return 1;
    if (!m->know) return 0;
    char a[2048];
    norm(answer, a, sizeof a);
    char *w[300];
    const int n = words(a, w, 300);
    int long_words = 0, known = 0;
    for (int i = 0; i < n; i++) {
        const int l = (int)strlen(w[i]);
        if (l < 4 || has_digit(w[i])) continue;
        long_words++;
        known += stopword(w[i]) || mlg_know_idf(m->know, w[i], l) >= 0.f;
    }
    /* the most common words are left out of the index (too common to search by), so a real
     * sentence still loses a few; gibberish loses nearly all */
    return long_words >= 3 && known * 2 < long_words;
}

/* A Resident Core's raw continuation -> the answer: its first line, without a
 * trailing half sentence. The evidence engine has the last word: an answer to
 * a question with evidence must rest on it, and a looping answer is not said
 * at all; the best sentences of the evidence are said instead. */
void mlg_mind_finish(mlg_mind *m, const mlg_thought *t, const char *raw, char *out, int cap) {
    char s[4096], q[1024] = "";
    copy(s, sizeof s, raw);
    char *nl = strchr(s, '\n');
    if (nl) *nl = 0;
    trim(s);
    if (starts(s, "M:")) memmove(s, s + 2, strlen(s) - 1), trim(s);
    const int n = (int)strlen(s);
    if (n > 80) {  /* cut a sentence left halfway at the length limit */
        int end = -1;
        for (int i = n - 1; i > n / 2; i--)
            if (strchr(".!?", s[i])) { end = i; break; }
        if (end > 0 && end < n - 1) s[end + 1] = 0;
    }
    const char *u = strstr(t->prompt, "U: ");  /* the question, as asked */
    if (u) {
        copy(q, sizeof q, u + 3);
        char *e = strstr(q, "\nM:");
        if (e) *e = 0;
    }
    int ok = strlen(s) >= 2 && !degenerate(s) && !gibberish(m, s);
    if (ok && t->evidence[0] && grounded(s, t->evidence, q) < 0.5) ok = 0;
    if (ok) { copy(out, cap, s), copy(m->spoke, sizeof m->spoke, "core"); return; }
    copy(m->spoke, sizeof m->spoke, t->evidence[0] ? "knowledge" : "trust");
    if (t->evidence[0]) {
        char best[700];
        extract(m, q, t->evidence, best, sizeof best);
        char in_context[1000];  /* a sentence is read within its article: "Marrocos" is said by the title */
        snprintf(in_context, sizeof in_context, "%s %s", starts(t->evidence_title, MLG_TITLE) ? "" : t->evidence_title, best);
        if (coverage(m, q, in_context) >= 0.7) snprintf(out, cap, "Pelo que eu sei: %s", best);
        else if (starts(t->evidence_title, MLG_TITLE) || !t->evidence_title[0])
            snprintf(out, cap, "Não achei exatamente isso. O que eu sei de mais perto: %s", best);
        else snprintf(out, cap, "Não achei exatamente isso. Sobre %s, eu sei que: %s", t->evidence_title, best);
    } else {
        copy(out, cap, "Não sei responder isso ainda. Me pergunte de outro jeito, ou me ensine: \"lembra que...\".");
    }
}

/* The answer from this body: its core when it has one, the evidence otherwise. */
int mlg_mind_speak(mlg_mind *m, const mlg_thought *t, char *out, int cap, mlg_on_byte cb, void *user, uint64_t seed) {
    if (t->final) { copy(out, cap, t->reply), copy(m->spoke, sizeof m->spoke, t->organ); return 1; }
    if (!m->core) {
        if (t->evidence[0]) { mlg_mind_finish(m, t, "", out, cap); return 0; }
        copy(m->spoke, sizeof m->spoke, "trust");
        /* what was asked, quoted back: an honest "not yet", never the same sentence twice in a row */
        const char *u = strstr(t->prompt, "U: ");
        char q[1024] = "", qq[96];
        if (u) copy(q, sizeof q, u + 3);
        char *e = strstr(q, "\nM:");
        if (e) *e = 0;
        quote(q, qq, sizeof qq);
        static const char *NOT_YET[2][3] = {
            {"Ainda não sei sobre \"%s\": meu conhecimento e meu Resident Core ainda estão no laboratório. Agora eu "
             "respondo contas, hora, memória e meu corpo.",
             "Sobre \"%s\" eu ainda não tenho nada: meus órgãos de conhecimento não chegaram. Se for importante, me ensine: "
             "\"lembra que...\".",
             "\"%s\": isso eu ainda não sei. Pergunte \"o que você sabe fazer?\" para ver o que já funciona neste corpo."},
            {"Sobre \"%s\" eu não achei nada no meu conhecimento. Meu Resident Core ainda está amadurecendo; quando chegar, "
             "eu converso sobre isso.",
             "Ainda não sei nada sobre \"%s\". Se for importante, me ensine: \"lembra que...\".",
             "Não encontrei \"%s\" no que eu sei. Por enquanto respondo contas, hora, memória, meu corpo e o que está no meu "
             "conhecimento."}};
        snprintf(out, cap, NOT_YET[m->know != NULL][m->turn % 3], qq);
        return 0;
    }
    uint8_t raw[400];
    const mlg_sample cfg = {0.6f, 0.9f, seed, 1, 4, 1};
    const int n = mlg_core_generate(m->core, (const uint8_t *)t->prompt, (int)strlen(t->prompt), raw, sizeof raw - 1,
                                    &cfg, cb, user);
    raw[n] = 0;
    mlg_mind_finish(m, t, (const char *)raw, out, cap);
    return 1;
}

/* The thought as JSON, for the HUD and the lab. */
int mlg_mind_thought_json(const mlg_mind *m, const mlg_thought *t, char *out, int cap) {
    char reply[4200], prompt[8400], ev[3200], evt[520];
    json_escape(t->reply, reply, sizeof reply);
    json_escape(t->prompt, prompt, sizeof prompt);
    json_escape(t->evidence, ev, sizeof ev);
    json_escape(t->evidence_title, evt, sizeof evt);
    int o = snprintf(out, cap, "{\"intent\":\"%s\",\"organ\":\"%s\",\"final\":%d,\"reply\":\"%s\",\"prompt\":\"%s\","
                               "\"evidence_title\":\"%s\",\"evidence\":\"%s\",\"score\":%.2f,\"probs\":{",
                     t->intent, t->organ, t->final, reply, prompt, evt, ev, t->evidence_score);
    for (int i = 0; m->router && i < t->n_intents && o < cap; i++)
        o += snprintf(out + o, cap - o, "%s\"%s\":%.4f", i ? "," : "", mlg_router_name(m->router, i), t->probs[i]);
    o += snprintf(out + o, cap - o, "}}");
    return o;
}
