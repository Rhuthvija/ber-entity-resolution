"""Text normalisation for business names and addresses.

Everything here is country-agnostic: it works for any Latin or Indic-script input,
so unseen countries (e.g. France in the test set) are handled by the same code path.
Native-script tokens are romanised with a dictionary learned from the TRAINING data
(learn_translit.py), falling back to anyascii for unseen tokens.
"""
import json
import os
import re
import unicodedata

from anyascii import anyascii

from common import ART

_TRANSLIT = None


def translit_dict():
    global _TRANSLIT
    if _TRANSLIT is None:
        p = os.path.join(ART, "translit.json")
        _TRANSLIT = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _TRANSLIT


ZW = re.compile(r"[​-‏﻿]")
NATIVE = re.compile(r"[ऀ-෿]")
NATIVE_TOK = re.compile(r"[ऀ-෿]+")

# ---------------------------------------------------------------- names
# Legal / company-form words, mapped to one canonical spelling.
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co",
    "opc": "opc", "pc": "pc", "pa": "pa",
    # French forms (no French training data: plain linguistic knowledge)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scop": "scop", "sca": "sca", "selarl": "selarl",
    # others
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
# multi-token legal forms after punctuation removal ("l l c", "s a s", "pvt ltd")
LEGAL_SEQ = [
    (("l", "l", "c"), "llc"), (("l", "l", "p"), "llp"), (("s", "a", "s"), "sas"),
    (("s", "a", "r", "l"), "sarl"), (("s", "a"), "sa"), (("p", "l", "c"), "plc"),
]
STOP = {"and", "the", "of", "a", "an", "s", "et", "de", "du", "des", "la", "le", "les", "l", "d", "en"}
HONORIFIC = {"mr", "mrs", "ms", "dr", "m s", "messrs", "mme", "mlle", "m"}
ALT_SPLIT = re.compile(r"\b(?:dba|d b a|doing business as|formerly|formerly known as|fka|f k a|aka|a k a|t a|trading as)\b")
DOMAIN = re.compile(r"\b([a-z0-9][a-z0-9\-]*)\s*\.\s*(com|net|org|biz|info|co\.in|in|fr|us|io|co)\b")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g",
                      "@": "a", "$": "s"})
# long legal words: also accept 1-2 character typos ("privte", "limted", "corporaton")
LEGAL_LONG = {"private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc", "company": "co"}
NONALNUM = re.compile(r"[^a-z0-9 ]+")
WS = re.compile(r"\s+")


def romanise(s):
    """NFKC, drop zero-width chars, map native-script words, then fold to ASCII."""
    if not s:
        return ""
    s = ZW.sub("", unicodedata.normalize("NFKC", s))
    if NATIVE.search(s):
        d = translit_dict()
        s = NATIVE_TOK.sub(lambda m: " " + d.get(m.group(0), anyascii(m.group(0))) + " ", s)
        # rejoin combining-mark fragments that the regex may have split
    return WS.sub(" ", anyascii(s).lower())


def _deleet(tok):
    """Undo OCR-style digit-for-letter noise ("appare1s", "6roup", "1ndia")."""
    if tok.isdigit() or sum(ch.isalpha() for ch in tok) < 2:
        return tok
    if tok[0] == "1":            # leading 1 is almost always a capital I (1nc, 1ndia)
        tok = "i" + tok[1:]
    return tok.translate(LEET)


_legal_cache = {}


def _legal_typo(tok):
    if tok in LEGAL or len(tok) < 6:
        return tok
    if tok not in _legal_cache:
        from rapidfuzz.distance import Levenshtein
        rep = tok
        for w, c in LEGAL_LONG.items():
            if abs(len(w) - len(tok)) <= 2 and Levenshtein.distance(tok, w) <= (1 if len(w) < 8 else 2):
                rep = w
                break
        _legal_cache[tok] = rep
    return _legal_cache[tok]


def _tokens(s):
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)      # l.l.c -> llc, s.a.s -> sas
    s = s.replace(".", "")                                  # pvt. -> pvt
    s = NONALNUM.sub(" ", s)
    toks = [_legal_typo(_deleet(t)) for t in s.split()]
    # merge single-letter legal sequences
    out, i = [], 0
    while i < len(toks):
        for seq, rep in LEGAL_SEQ:
            if tuple(toks[i:i + len(seq)]) == seq:
                out.append(rep)
                i += len(seq)
                break
        else:
            out.append(toks[i])
            i += 1
    return out


ID_TAG = re.compile(r"\b(?:id|ref|reg)\s*(?:no)?\s*[:#.]?\s*\d+")


def _core(toks):
    # long pure-digit tokens in a name are ids / phone numbers, never part of the business name
    core = [t for t in toks if t not in LEGAL and t not in STOP and t not in HONORIFIC
            and not (t.isdigit() and len(t) >= 5)]
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    return core, legal


def norm_name(raw):
    """Return dict with full/core/alt name strings, legal forms and flags."""
    s = ID_TAG.sub(" ", romanise(raw))            # "(ID: 58156)" noise
    is_dom = 0
    m = DOMAIN.search(s)
    if m and len(s.split()) <= 2:
        s = m.group(1).replace("-", " ")
        is_dom = 1
    alt = ""
    parts = ALT_SPLIT.split(NONALNUM.sub(" ", s.replace(".", "").replace("&", " and ")), maxsplit=1)
    if len(parts) == 2:
        s, alt = parts[0], parts[1]
    toks = _tokens(s)
    core, legal = _core(toks)
    alt_core = _core(_tokens(alt))[0] if alt else []
    return {
        "n_full": " ".join(toks),
        "n_core": " ".join(core),
        "n_alt": " ".join(alt_core),
        "n_legal": " ".join(legal),
        "n_dom": is_dom,
    }


# ---------------------------------------------------------------- addresses
ADDR_ABBR = {
    # street types (US / India)
    "street": "st", "st": "st", "str": "st", "saint": "st", "avenue": "ave", "ave": "ave", "av": "ave",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "court": "ct", "ct": "ct", "circle": "cir",
    "cir": "cir", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "lane": "ln", "ln": "ln",
    "place": "pl", "pl": "pl", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste", "floor": "fl", "flr": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg", "nagar": "nagar", "marg": "marg", "sector": "sec", "sec": "sec",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt", "township": "twp",
    "twp": "twp", "crossing": "xing", "cross": "cross", "main": "main", "opposite": "opp", "opp": "opp",
    # French street types
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "impasse": "imp", "imp": "imp",
    "chemin": "ch", "ch": "ch", "route": "rte", "rte": "rte", "quai": "quai", "cite": "cite",
    "residence": "res", "res": "res", "faubourg": "fbg", "fbg": "fbg", "sainte": "ste",
}
# filler tokens that carry no location information
ADDR_DROP = {"no", "number", "num", "h", "hno", "house", "door", "unit", "near", "nr", "behind", "c", "o",
             "null", "none", "na", "the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l",
             "bis", "ter", "po", "box", "city"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld", "andaman and nicobar islands": "an", "telengana": "ts",
}
_STATE_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
    for c, d in (("US", US_STATES), ("India", IN_STATES))
}
ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
# mailbox / unit designators and their value: noise that one source adds and another drops
UNIT = re.compile(r"\b(?:pmb|p\s*o\s*box|po\s*box|box|unit|ste|suite|apt|apartment|desk|fl|floor|room|rm)\b\s*#?\s*[a-z]?\d*[a-z]?\b")
RANGE = re.compile(r"\b(\d+)\s*-\s*(\d+)\b")


def _collapse_range(m):
    a, b = m.group(1), m.group(2)
    # "8400-8404" / "506-510": a street-number range -> keep the first number
    if len(a) == len(b) and int(b) > int(a) and int(b) - int(a) < 50:
        return a
    return m.group(0)


def norm_addr(raw, country):
    s = romanise(raw)
    if not s.strip():
        return {"a_norm": "", "a_nums": "", "a_state": ""}
    s = s.replace("&", " and ")
    state = ""
    st_map = US_STATES if country == "US" else IN_STATES if country == "India" else None
    comps = [c.strip() for c in s.split(",")]
    if st_map is not None:
        abbrs = set(st_map.values())
        rx = _STATE_RE[country]
        for c in comps:
            cc = WS.sub(" ", NONALNUM.sub(" ", c)).strip()
            if cc in st_map:
                state = st_map[cc]
            elif cc in abbrs and len(comps) > 1:
                state = cc
        s = rx.sub(lambda m: st_map[m.group(1)], s)
    s = ORD.sub(r"\1", s)
    s = RANGE.sub(_collapse_range, s)
    s = s.replace("#", " ")
    s = UNIT.sub(" ", s)
    s = s.replace(".", " ").replace("'", " ")
    s = NONALNUM.sub(" ", s)
    toks = []
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        if t in ADDR_DROP:
            continue
        toks.append(t)
    nums = re.findall(r"\d+", " ".join(toks))
    nums = [n.lstrip("0") or "0" for n in nums]
    return {"a_norm": " ".join(toks), "a_nums": " ".join(nums), "a_state": state}


if __name__ == "__main__":
    tests = [
        ("Espinoza & Davis Industrials L.L.C.", "5425 152ND AVENUE, RAMSEY, MN", "US"),
        ("Onyxveo formerly Espinoza & Davis Industrials LLC", "1425 152nd Ave, Anoka CITY, Minnesota", "US"),
        ("ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "G.S.RESIDENCY, SITE NO.8, BANGALORE, ಕರ್ನಾಟಕ", "India"),
        ("Avikor DBA: Life Investments", "Block D-6-86 Office No 212, Thane, MH", "India"),
        ("lifeinvestments.com", "4/792., SECTOR 4, VIKAS NAGAR, LUCKNOW, उत्तर प्रदेश", "India"),
        ("Lucknow Appare1s", "2 Muldowney Cir, # Apartment B, Poughkeepsie, New York", "US"),
        ("Fractales Amis Groupe S.A.S", "23 R. Icmre, La Teste-de-buch, Gironde", "France"),
        ("*** Blue Hypnósis", "506-510 BOONE ST, PO BOX 6058, OLNEY, IL", "US"),
        ("Sri MAGPPIE PARK PVT LTD", "", "India"),
        ("Krauss's Pizza Corporation (ID: 58156)", "6601-6605 THURSTON AVE, PMB 2727, SAINT LOUIS, MO", "US"),
        ("Krauss's Pizza 9876543210", "6601 Thurston Avenue, Fl 0, Unit UNIT 414, Saint Louis, MO", "US"),
        ("X", "Office No 212, 1St Floor, 57-14-24 W Rd, 4/792, 12-90 Colony, Thane, MH", "India"),
    ]
    for n, a, c in tests:
        print(norm_name(n), norm_addr(a, c))
