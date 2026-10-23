/* Text as the router and the retrieval expert see it. mlgsub/text.py does
 * exactly the same, byte for byte, so the index and the router trained in
 * the lab read text the way the headset does. */
#include "mlg.h"

/* U+00C0..U+00FF (UTF-8 C3 80..C3 BF) folded to ASCII; ' ' means a separator */
static const char LATIN1[] = "aaaaaaaceeeeiiiidnooooo ouuuuy saaaaaaaceeeeiiiidnooooo ouuuuy y";

int mlg_normalize(const char *in, int n, char *out, int cap) {
    const unsigned char *s = (const unsigned char *)in;
    int o = 0, space = 1;
#define PUT(ch) ((o < cap ? (void)(out[o] = (char)(ch)) : (void)0), o++)
    for (int i = 0; i < n; i++) {
        unsigned char b = s[i];
        if (b < 0x80) {
            if (b >= 'A' && b <= 'Z') b += 32;
            if ((b >= 'a' && b <= 'z') || (b >= '0' && b <= '9')) PUT(b), space = 0;
            else if (!space) PUT(' '), space = 1;
        } else if (b == 0xC3 && i + 1 < n && s[i + 1] >= 0x80 && s[i + 1] <= 0xBF) {
            const char f = LATIN1[s[++i] - 0x80];
            if (f != ' ') PUT(f), space = 0;
            else if (!space) PUT(' '), space = 1;
        } else if (b == 0xC2 && i + 1 < n && s[i + 1] >= 0x80 && s[i + 1] <= 0xBF) {
            /* U+0080..U+00BF: no-break space, quotes, degrees... ordinals become letters */
            const unsigned char c = s[++i];
            if (c == 0xAA || c == 0xBA) PUT(c == 0xAA ? 'a' : 'o'), space = 0;
            else if (!space) PUT(' '), space = 1;
        } else if (b == 0xE2 && i + 2 < n && s[i + 1] == 0x80) {
            i += 2;  /* U+2000..U+203F: dashes, curly quotes, ellipsis */
            if (!space) PUT(' '), space = 1;
        } else {  /* any other character stays whole, as part of a word */
            int len = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
            for (int j = 0; j < len && i + j < n; j++) PUT(s[i + j]);
            i += len - 1;
            space = 0;
        }
    }
    if (o > 0 && o <= cap && out[o - 1] == ' ') o--;
#undef PUT
    return o < cap ? o : cap;
}

uint32_t mlg_hash(const char *s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 16777619u;
    return h;
}
