"""Text as the router and the retrieval expert see it: engine/mlg_text.c, in
Python, byte for byte (tests/test_engine.py holds the two to the same output)."""

LATIN1 = b"aaaaaaaceeeeiiiidnooooo ouuuuy saaaaaaaceeeeiiiidnooooo ouuuuy y"


def normalize(text):
    """Lowercase, accents folded, non-alphanumerics as single spaces. Bytes in, bytes out."""
    s = text.encode("utf-8") if isinstance(text, str) else text
    out, space, i, n = bytearray(), True, 0, len(s)
    while i < n:
        b = s[i]
        if b < 0x80:
            if 65 <= b <= 90:
                b += 32
            if 97 <= b <= 122 or 48 <= b <= 57:
                out.append(b)
                space = False
            elif not space:
                out.append(32)
                space = True
        elif b == 0xC3 and i + 1 < n and 0x80 <= s[i + 1] <= 0xBF:
            i += 1
            f = LATIN1[s[i] - 0x80]
            if f != 32:
                out.append(f)
                space = False
            elif not space:
                out.append(32)
                space = True
        elif b == 0xC2 and i + 1 < n and 0x80 <= s[i + 1] <= 0xBF:
            i += 1
            if s[i] in (0xAA, 0xBA):
                out.append(97 if s[i] == 0xAA else 111)
                space = False
            elif not space:
                out.append(32)
                space = True
        elif b == 0xE2 and i + 2 < n and s[i + 1] == 0x80:
            i += 2
            if not space:
                out.append(32)
                space = True
        else:
            length = 4 if b >= 0xF0 else 3 if b >= 0xE0 else 2 if b >= 0xC0 else 1
            out += s[i:i + length]
            i += length - 1
            space = False
        i += 1
    if out and out[-1] == 32:
        out.pop()
    return bytes(out)


def words(text):
    n = normalize(text)
    return n.split(b" ") if n else []


def fnv1a(data, basis=2166136261):
    h = basis
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h
