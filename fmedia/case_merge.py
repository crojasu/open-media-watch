"""
case_merge.py
-------------
Pure matcher that PROPOSES merges of duplicate Cases. It never
touches a database: callers hand it `CaseRecord`s and get `MergeGroup`s back.

Owner's rules
  1. The same victim name reported in two countries is ONE case (cases are
     created with the OUTLET's country, so a German outlet covering a London
     killing produced a second Case).
  2. An unnamed victim in the same town with close dates is very probably the
     same case.

Confidence tiers
  A  auto-safe candidate: same strict full name (>= 2 tokens on both sides,
     reuses cases._normalize_name/_name_match), dates inside A window, any
     country (unless BOTH sides have a known, different incident country).
  B  needs human review: (a) name equal but no usable dates or conflicting known
     incident countries; (b) unnamed/placeholder victim + same country + same
     specific town + dates within 7 days (+ compatible type).
     (c) PLACELESS variant: unnamed + no town on either side + same country +
     same real incident date (<= 1 day) + same known weapon + text evidence
     (>= 3 shared word stems [first 5 letters] in titles/summaries, Jaccard >= 0.12, or
     equal known ages). Date + weapon alone is never enough -> Tier C.
  C  weak, never automatic: same country + same town + same year, at least one
     side unnamed.

 additions (after labelling 30 Tier-A and 45 Tier-B production groups,
A 28/30 and B 41/45 correct): a name match is Tier A only when the token sets
are equal or one full name contains the other and the shorter side is not just
common given names ("Ana María" vs "Ana María Sampleton" and "Xaviera" vs
"Javiera Ejemplo Sample" were the two A errors -> now Tier B); the same full
perpetrator name is a Tier-A edge inside PERP_WINDOW_DAYS; in BIG_CITIES the
unnamed town rule needs <= 1 day plus text evidence (3 of the 4 B errors were
"London/Puebla, 6-7 days apart").

Hard guards (every tier): fatal vs non-fatal types never merge; known victim
ages differing by more than 2 never merge; two different full names never merge.
Weapon text is NOT compared (free text, too noisy to be a guard).

Dates. `incident_date` is only set on ~3% of cases, so the reference date falls
back to the first-published date. A window: both dates are real incident dates ->
45 days (extraction noise); otherwise 730 days, because the fallback is the
publication date and trial / anniversary coverage arrives months later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Iterable, Optional

from .cases import _name_match, _normalize, _normalize_name

A_WINDOW_EXACT_DAYS = 45
A_WINDOW_FALLBACK_DAYS = 730
B_WINDOW_DAYS = 7
MAX_AGE_GAP = 2
MIN_NAME_TOKENS = 2
_BLOCK_CAP = 400  # skip name-token blocks larger than this (stop-word-like tokens)

# internal country code <-> ISO aliases (GB is the ISO spelling of our "UK")
_COUNTRY_ALIAS = {"GB": "UK"}

_PLACEHOLDERS = {
    "desconocido", "desconocida", "unknown", "n a", "na", "none", "null", "sin nombre",
    "sin identificar", "no identificada", "no identificado", "unnamed", "unidentified",
    "unidentified woman", "unnamed woman", "mujer", "woman", "nn", "n n", "anonima", "anonimo",
    "victima", "victim", "not named", "not identified", "no disponible", "por identificar",
    # seen in production: German / Spanish variants and generic nouns
    "unbekannt", "unbekannte", "unbekannte frau", "frau", "eine frau", "una mujer",
    "no especificado", "no especificada", "mujer no identificada", "una mujer no identificada",
    "no consta", "no se informa", "reservado", "reserved", "heiress", "wife", "esposa", "mother", "madre",
}
_PLACEHOLDER_PREFIXES = ("unidentified ", "unnamed ", "desconocid", "sin identificar", "unbekannt",
                         "una mujer", "mujer de ", "woman in her", "frau ", "no identificad")

# Given names so common that "<given> <given>" alone (e.g. "Ana María") must not
# be treated as identifying a person: a subset match on such a name is Tier B.
_COMMON_GIVEN = {
    "ana", "maria", "jose", "juan", "luis", "carmen", "laura", "rosa", "isabel", "marta", "paula",
    "lucia", "sofia", "camila", "valentina", "andrea", "daniela", "gabriela", "patricia", "claudia",
    "monica", "sandra", "sarah", "sara", "emma", "anna", "hannah", "jessica", "jennifer", "michelle",
    "amy", "laura", "lisa", "julia", "sophie", "sophia", "katie", "emily", "charlotte", "mary", "jane",
    "elizabeth", "elena", "cristina", "natalia", "karla", "jenny", "alma", "diana", "romina", "raquel",
    "valeria", "eugenia", "karen", "veronica", "fatima", "milena", "melanie", "lidia",
    # male given names (perpetrators)
    "angel", "miguel", "carlos", "pedro", "antonio", "manuel", "francisco", "david", "daniel", "jorge",
    "eduardo", "alejandro", "javier", "john", "james", "michael", "paul", "mark", "andrew", "thomas",
    "robert", "peter", "hans", "jose", "luis", "juan", "ricardo", "fernando", "sergio", "diego",
}

# Towns where "same town + a few days" is weak evidence by itself (large cities):
# Tier B there needs same-day (<= 1 day) coverage plus text evidence or equal age.
BIG_CITIES = {
    "london", "londres", "birmingham", "manchester", "glasgow",
    "berlin", "hamburg", "munchen", "munich", "koln", "cologne", "frankfurt",
    "madrid", "barcelona", "valencia", "sevilla", "zaragoza", "malaga", "santiago", "buenos aires",
    "caba", "capital federal", "ciudad de mexico", "cdmx", "mexico city", "guadalajara", "monterrey",
    "puebla", "tijuana", "ecatepec", "bogota", "lima", "sao paulo", "rio de janeiro", "new york",
    "los angeles", "chicago", "paris", "rome", "roma", "istanbul", "delhi", "mumbai", "karachi",
    "ciudad juarez", "juarez", "edomex", "estado de mexico", "nezahualcoyotl", "sydney", "melbourne",
}
BIG_CITY_WINDOW_DAYS = 1

PERP_WINDOW_DAYS = 400  # perpetrator name match: Tier A only inside this window
_PERP_PLACEHOLDERS = {
    "husband", "wife", "partner", "ex", "boyfriend", "ex partner", "ex boyfriend", "esposo", "marido",
    "pareja", "expareja", "ex pareja", "novio", "exnovio", "ex novio", "ehemann", "partner", "freund",
    "ex freund", "tatverdachtiger", "unknown", "desconocido", "n", "sujeto", "hombre", "man", "mann",
}
_GENERIC_PLACES = {
    "", "unknown", "desconocido", "n a", "na", "uk", "united kingdom", "england", "scotland",
    "wales", "northern ireland", "great britain", "gb", "chile", "argentina", "mexico", "espana",
    "spain", "germany", "deutschland", "home", "house", "her home", "domicilio", "casa",
    "vivienda", "portugal", "fiji", "france", "francia", "italy", "italia", "india", "usa", "united states",
    "ireland", "australia", "brasil", "brazil", "colombia", "peru", "ecuador", "bolivia", "venezuela",
    "lancashire", "yorkshire", "kent", "essex", "surrey", "devon", "cornwall", "unterfranken", "oberbayern",
    "bayern", "bavaria", "nordrhein westfalen", "river thames", "thames", "andalucia", "cataluna", "galicia",
    "country", "pais", "region", "ciudad", "city", "town",
}

from .taxonomy import NON_FATAL_TYPES as _NON_FATAL_TYPES  # noqa: E402

PLACELESS_MIN_SHARED_WORDS = 3
PLACELESS_MIN_JACCARD = 0.12
_STOPWORDS = set("""the a an and or of in on at to for with by from after as is was were be been has have had
her his their she he they it its that this who whom while after before into over out up down off his hers
news video pictured court el la los las un una de del en y o con por para que se su sus al lo mujer hombre
die der das und mit von zu im ein eine den dem nach""".split())

TIER_ORDER = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class CaseRecord:
    id: int
    country: str
    victim_name: Optional[str] = None
    victim_age: Optional[int] = None
    incident_location: Optional[str] = None
    incident_region: Optional[str] = None
    incident_date: Optional[date] = None
    first_published: Optional[date] = None
    event_type: Optional[str] = None
    perpetrator_name: Optional[str] = None
    article_count: int = 1
    weapon: Optional[str] = None
    # titles / summaries of the case's articles (used only as extra evidence for
    # placeless Tier B: never merged on date + weapon alone)
    texts: tuple = ()
    # incident_country of each of the case's articles (None/"" entries ignored)
    article_incident_countries: tuple = ()


@dataclass(frozen=True)
class MergeGroup:
    tier: str
    survivor_id: int
    member_ids: tuple          # sorted ids, survivor included
    country: str               # effective country of the merged case
    reasons: tuple = field(default_factory=tuple)


# ── primitives ───────────────────────────────────────────────────────────────

def _canon_country(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().upper()
    return _COUNTRY_ALIAS.get(code, code)


def effective_country(rec: CaseRecord) -> tuple:
    """(country, known). Majority incident country of the case's articles
    (GB==UK); ties prefer the outlet country; no info -> (outlet, False)."""
    outlet = _canon_country(rec.country)
    votes = Counter(c for c in (_canon_country(x) for x in rec.article_incident_countries) if c)
    if not votes:
        return outlet, False
    top = max(votes.values())
    leaders = sorted(c for c, n in votes.items() if n == top)
    return (outlet if outlet in leaders else leaders[0]), True


def is_placeholder_name(name: Optional[str]) -> bool:
    n = _normalize(name or "")
    if not n or n in _PLACEHOLDERS:
        return True
    return n.startswith(_PLACEHOLDER_PREFIXES)


def _tokens(name: Optional[str]) -> list:
    return _normalize_name(name or "").split()


def _is_full_name(name: Optional[str]) -> bool:
    """>= 2 tokens, none shorter than 2 chars (drops Mexican "Febronio N" style anonymisation)."""
    if is_placeholder_name(name):
        return False
    toks = _tokens(name)
    return len(toks) >= MIN_NAME_TOKENS and all(len(t) >= 2 for t in toks)


is_full_name = _is_full_name  # public alias for the ingestion linker


def name_match_strength(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """'exact'  -> same token set, or one full name contained in the other where the
                  shorter side is not just common given names (Tier A evidence);
       'fuzzy'  -> typo-level similarity or a subset of only common given names
                  ("Ana María" vs "Ana María Sampleton", "Xaviera" vs "Javiera"
                  Rojas were production false merges) -> Tier B at most;
       None     -> different people."""
    if not (_is_full_name(a) and _is_full_name(b)):
        return None
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if ta == tb:
        return "exact"
    if ta <= tb or tb <= ta:
        shorter = ta if len(ta) <= len(tb) else tb
        return "exact" if any(t not in _COMMON_GIVEN for t in shorter) else "fuzzy"
    if len(ta & tb) >= 2:
        return "exact" if len(ta & tb) >= 2 and len(ta ^ tb) <= 2 else "fuzzy"
    if SequenceMatcher(None, _normalize_name(a), _normalize_name(b)).ratio() > 0.9:
        return "fuzzy"
    return None


def _strict_name_match(a: Optional[str], b: Optional[str]) -> bool:
    """Token-aware "same person" test (exact or fuzzy); no character substring hits ('ana' in 'diana')."""
    return name_match_strength(a, b) is not None


def _is_full_perp_name(name: Optional[str]) -> bool:
    n = _normalize_name(name or "")
    if not n or n in _PERP_PLACEHOLDERS:
        return False
    toks = n.split()
    return len(toks) >= MIN_NAME_TOKENS and all(len(t) >= 2 for t in toks)


def _perp_match(a: Optional[str], b: Optional[str]) -> bool:
    """Same perpetrator: equal token sets (or containment sharing >= 2 tokens) and
    at least one token that is not a common given name ('Luis Ángel' alone is not
    an identity)."""
    if not (_is_full_perp_name(a) and _is_full_perp_name(b)):
        return False
    ta, tb = set(_normalize_name(a).split()), set(_normalize_name(b).split())
    if not (ta == tb or ((ta <= tb or tb <= ta) and len(ta & tb) >= 2)):
        return False
    return any(t not in _COMMON_GIVEN for t in ta & tb)


# Vocabulary so generic in crime headlines that sharing it proves nothing
# (first-5-letter stems, see _content_words).
GENERIC_STEMS = {
    "woman", "women", "man", "murde", "murder", "charg", "arres", "found", "dead", "death", "kille",
    "killi", "stabb", "polic", "body", "court", "home", "house", "after", "mujer", "hombr", "asesi",
    "femin", "femic", "muert", "hallan", "encue", "cuerp", "deten", "frau", "mann", "getot",
    "tot", "leich", "mord", "festg", "pareja", "expar", "partn", "wife", "husba", "mother", "mum",
    "year", "years", "old", "jahre", "anos", "london", "londr", "berli", "madri", "santi", "mexic",
}


def is_big_city(town: Optional[str]) -> bool:
    return bool(town) and _normalize(town) in BIG_CITIES


def ref_date(rec: CaseRecord) -> tuple:
    """(date, is_real_incident_date). An incident_date that is impossible next to
    the first coverage (incident > 30 days AFTER first publication, or > 365 days
    before it) is an extraction error (seen: 2017 articles dated 2024-12-25) and
    is ignored in favour of the publication date."""
    if rec.incident_date:
        fp = rec.first_published
        if fp is None or -30 <= (fp - rec.incident_date).days <= 365:
            return rec.incident_date, True
        return fp, False
    if rec.first_published:
        return rec.first_published, False
    return None, False


def _town(rec: CaseRecord) -> Optional[str]:
    loc = rec.incident_location
    if not loc:
        return None
    t = _normalize(str(loc).split(",")[0])
    return None if t in _GENERIC_PLACES or len(t) < 3 else t


def _is_fatal(t: Optional[str]) -> Optional[bool]:
    if not t:
        return None
    return t not in _NON_FATAL_TYPES


def _guards_ok(a: CaseRecord, b: CaseRecord) -> bool:
    fa, fb = _is_fatal(a.event_type), _is_fatal(b.event_type)
    if fa is not None and fb is not None and fa != fb:
        return False
    if a.victim_age is not None and b.victim_age is not None \
            and abs(a.victim_age - b.victim_age) > MAX_AGE_GAP:
        return False
    if _is_full_name(a.victim_name) and _is_full_name(b.victim_name) \
            and not _strict_name_match(a.victim_name, b.victim_name):
        return False
    return True


def _span_ok(a: CaseRecord, b: CaseRecord) -> bool:
    """Stop chaining: two UNNAMED cases in one group must themselves be within the B window."""
    if is_placeholder_name(a.victim_name) and is_placeholder_name(b.victim_name) \
            and not _perp_match(a.perpetrator_name, b.perpetrator_name):
        dd = _days(a, b)
        return dd is None or dd[0] <= B_WINDOW_DAYS
    return True


def _days(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    (da, ea), (db_, eb) = ref_date(a), ref_date(b)
    if da is None or db_ is None:
        return None
    return abs((da - db_).days), (ea and eb)


# ── pair classification ──────────────────────────────────────────────────────

def _name_edge(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    """Tier-A/B edge from an identical full name, else None."""
    strength = name_match_strength(a.victim_name, b.victim_name)
    if strength is None:
        return None
    dd = _days(a, b)
    if dd is None:
        return "B", "same full name but no usable dates"
    days, both_exact = dd
    window = A_WINDOW_EXACT_DAYS if both_exact else A_WINDOW_FALLBACK_DAYS
    if days > window:
        return None
    (ca, ka), (cb, kb) = effective_country(a), effective_country(b)
    if ka and kb and ca != cb:
        return "B", f"same full name, different incident countries ({ca}/{cb}), {days}d apart"
    if strength == "fuzzy":
        return "B", f"similar name '{a.victim_name}' / '{b.victim_name}', {days}d apart"
    cross = f", outlets {a.country}/{b.country}" if a.country != b.country else ""
    return "A", f"same full name, {days}d apart{cross}"


def _perp_edge(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    """Tier-A/B edge from the same full perpetrator name (12 production cases shared
    'John Sample'). Victim-name conflicts are rejected by _guards_ok."""
    if not _perp_match(a.perpetrator_name, b.perpetrator_name):
        return None
    dd = _days(a, b)
    if dd is None:
        return "B", "same perpetrator name but no usable dates"
    days = dd[0]
    if days > A_WINDOW_FALLBACK_DAYS:
        return None
    (ca, ka), (cb, kb) = effective_country(a), effective_country(b)
    if ka and kb and ca != cb:
        return "B", f"same perpetrator '{a.perpetrator_name}', different incident countries ({ca}/{cb})"
    if days > PERP_WINDOW_DAYS:
        return "B", f"same perpetrator '{a.perpetrator_name}', {days}d apart"
    return "A", f"same perpetrator '{a.perpetrator_name}', {days}d apart"


def _town_edge(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    """Tier-B edge for unnamed victim + same country/town/close dates. In a big
    city the window shrinks to BIG_CITY_WINDOW_DAYS and text evidence (or an
    equal known age) is required: 'London, 7 days apart' was a production false merge."""
    ua, ub = is_placeholder_name(a.victim_name), is_placeholder_name(b.victim_name)
    if not (ua or ub):
        return None
    ta, tb = _town(a), _town(b)
    if not ta or ta != tb or effective_country(a)[0] != effective_country(b)[0]:
        return None
    dd = _days(a, b)
    if dd is None or dd[0] > B_WINDOW_DAYS:
        return None
    if a.event_type and b.event_type and a.event_type != b.event_type:
        return None
    kind = "both unnamed" if (ua and ub) else "one unnamed"
    if is_big_city(ta):
        if dd[0] > BIG_CITY_WINDOW_DAYS:
            return None
        ev = _distinctive_text_evidence(a, b)
        if ev is None:
            return None
        return "B", f"{kind}, big city '{ta}', {dd[0]}d apart, {ev}"
    return "B", f"{kind}, same town '{ta}', {dd[0]}d apart"


def _content_words(rec: CaseRecord) -> set:
    words = set()
    for t in rec.texts:
        words.update(w[:5] for w in _normalize(t or "").split() if len(w) >= 4 and w not in _STOPWORDS)
    return words


def _text_evidence(a: CaseRecord, b: CaseRecord) -> Optional[str]:
    if a.victim_age is not None and a.victim_age == b.victim_age:
        return f"equal age {a.victim_age}"
    wa, wb = _content_words(a), _content_words(b)
    shared = wa & wb
    if wa and wb and len(shared) >= PLACELESS_MIN_SHARED_WORDS \
            and len(shared) / len(wa | wb) >= PLACELESS_MIN_JACCARD:
        return f"text overlap ({len(shared)} shared words)"
    return None


def _distinctive_text_evidence(a: CaseRecord, b: CaseRecord) -> Optional[str]:
    """Like _text_evidence but ignoring GENERIC_STEMS: for big cities, 'woman
    stabbed to death in London' shared by two headlines proves nothing."""
    if a.victim_age is not None and a.victim_age == b.victim_age:
        return f"equal age {a.victim_age}"
    wa, wb = _content_words(a) - GENERIC_STEMS, _content_words(b) - GENERIC_STEMS
    shared = wa & wb
    if wa and wb and len(shared) >= PLACELESS_MIN_SHARED_WORDS \
            and len(shared) / len(wa | wb) >= PLACELESS_MIN_JACCARD:
        return f"distinctive text overlap ({len(shared)} shared words)"
    return None


def _placeless_ok(a: CaseRecord, b: CaseRecord) -> Optional[int]:
    """Days apart if both unnamed, placeless, same country, real dates <=1d, same weapon."""
    if not (is_placeholder_name(a.victim_name) and is_placeholder_name(b.victim_name)):
        return None
    if _town(a) or _town(b):
        return None
    if effective_country(a)[0] != effective_country(b)[0]:
        return None
    (da, ea), (db_, eb) = ref_date(a), ref_date(b)
    if not (ea and eb):
        return None
    days = abs((da - db_).days)
    wa, wb = _normalize(a.weapon or ""), _normalize(b.weapon or "")
    if days > 1 or not wa or wa != wb:
        return None
    return days


def _placeless_edge(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    days = _placeless_ok(a, b)
    if days is None:
        return None
    ev = _text_evidence(a, b)
    if ev is None:
        return None
    return "B", f"placeless: unnamed, same day (+{days}d), same weapon '{a.weapon}', {ev}"


def _weak_edge(a: CaseRecord, b: CaseRecord) -> Optional[tuple]:
    if _placeless_ok(a, b) is not None:
        return "C", "placeless: same day + weapon only, no text evidence"
    if not (is_placeholder_name(a.victim_name) or is_placeholder_name(b.victim_name)):
        return None
    ta, tb = _town(a), _town(b)
    if not ta or ta != tb or effective_country(a)[0] != effective_country(b)[0]:
        return None
    (da, _), (db_, _) = ref_date(a), ref_date(b)
    if da is None or db_ is None or da.year != db_.year:
        return None
    return "C", f"same town '{ta}', same year {da.year} only"


# ── candidate generation (blocking) ──────────────────────────────────────────

def _name_pairs(recs: list) -> Iterable:
    blocks = defaultdict(list)
    for r in recs:
        if _is_full_name(r.victim_name):
            for t in set(_tokens(r.victim_name)):
                if len(t) >= 3:
                    blocks[t].append(r)
    seen = set()
    for members in blocks.values():
        if len(members) > _BLOCK_CAP:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = (min(a.id, b.id), max(a.id, b.id))
                if key not in seen:
                    seen.add(key)
                    yield a, b


def _perp_pairs(recs: list) -> Iterable:
    blocks = defaultdict(list)
    for r in recs:
        if _is_full_perp_name(r.perpetrator_name):
            for t in set(_normalize_name(r.perpetrator_name).split()):
                if len(t) >= 3:
                    blocks[t].append(r)
    seen = set()
    for members in blocks.values():
        if len(members) > _BLOCK_CAP:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = (min(a.id, b.id), max(a.id, b.id))
                if key not in seen:
                    seen.add(key)
                    yield a, b


def _town_blocks(recs: list) -> dict:
    blocks = defaultdict(list)
    for r in recs:
        t = _town(r)
        if t:
            blocks[(effective_country(r)[0], t)].append(r)
    return blocks


def _town_pairs_b(recs: list) -> Iterable:
    for members in _town_blocks(recs).values():
        dated = sorted((m for m in members if ref_date(m)[0]), key=lambda m: (ref_date(m)[0], m.id))
        for i, a in enumerate(dated):
            for b in dated[i + 1:]:
                if (ref_date(b)[0] - ref_date(a)[0]).days > B_WINDOW_DAYS:
                    break
                yield a, b


def _placeless_blocks(recs: list) -> Iterable:
    blocks = defaultdict(list)
    for r in recs:
        if is_placeholder_name(r.victim_name) and not _town(r) and ref_date(r)[1] and r.weapon:
            blocks[(effective_country(r)[0], ref_date(r)[0], _normalize(r.weapon))].append(r)
    return blocks


def _placeless_pairs(recs: list) -> Iterable:
    # same-day blocks plus +1 day neighbours
    blocks = _placeless_blocks(recs)
    for (c, d, w), members in blocks.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                yield a, b
        nxt = blocks.get((c, d.fromordinal(d.toordinal() + 1), w), [])
        for a in members:
            for b in nxt:
                yield a, b


def _town_pairs_c(recs: list) -> Iterable:
    for members in _town_blocks(recs).values():
        by_year = defaultdict(list)
        for m in members:
            d = ref_date(m)[0]
            if d:
                by_year[d.year].append(m)
        for ys in by_year.values():
            for i, a in enumerate(ys):
                for b in ys[i + 1:]:
                    yield a, b


# ── grouping ─────────────────────────────────────────────────────────────────

def _survivor(members: list) -> CaseRecord:
    return sorted(members, key=lambda r: (-(r.article_count or 0), r.id))[0]


def _make_group(tier: str, members: list, reasons: list) -> MergeGroup:
    surv = _survivor(members)
    counts = Counter()
    for m in members:
        counts.update(c for c in (_canon_country(x) for x in m.article_incident_countries) if c)
    if counts:
        top = max(counts.values())
        leaders = sorted(c for c, n in counts.items() if n == top)
        sc = _canon_country(surv.country)
        country = sc if sc in leaders else leaders[0]
    else:
        country = _canon_country(surv.country)
    return MergeGroup(tier=tier, survivor_id=surv.id,
                      member_ids=tuple(sorted(m.id for m in members)),
                      country=country, reasons=tuple(sorted(set(reasons))))


def propose_merges(records: Iterable) -> list:
    """Return MergeGroups (tiers A/B = connected groups, C = pairs), deterministic."""
    recs = sorted(records, key=lambda r: r.id)
    edges = []
    for a, b in _name_pairs(recs):
        e = _name_edge(a, b)
        if e and _guards_ok(a, b):
            edges.append((TIER_ORDER[e[0]], a.id, b.id, a, b, e))
    for a, b in _perp_pairs(recs):
        e = _perp_edge(a, b)
        if e and _guards_ok(a, b):
            edges.append((TIER_ORDER[e[0]], min(a.id, b.id), max(a.id, b.id), a, b, e))
    for a, b in _town_pairs_b(recs):
        e = _town_edge(a, b)
        if e and _guards_ok(a, b):
            edges.append((TIER_ORDER[e[0]], min(a.id, b.id), max(a.id, b.id), a, b, e))
    for a, b in _placeless_pairs(recs):
        e = _placeless_edge(a, b)
        if e and _guards_ok(a, b):
            edges.append((TIER_ORDER[e[0]], min(a.id, b.id), max(a.id, b.id), a, b, e))
    edges.sort(key=lambda x: x[:3])

    comp = {r.id: [r] for r in recs}          # id -> shared member list
    worst = {r.id: "A" for r in recs}
    why = {r.id: [] for r in recs}
    merged_ids = set()
    for _, _, _, a, b, (tier, reason) in edges:
        ca, cb = comp[a.id], comp[b.id]
        if ca is cb:
            continue
        if not all(_guards_ok(x, y) and _span_ok(x, y) for x in ca for y in cb):
            continue
        ra, rb = ca[0].id, cb[0].id
        new = ca + cb
        tier_w = max(worst[ra], worst[rb], tier, key=lambda t: TIER_ORDER[t])
        reasons = why[ra] + why[rb] + [reason]
        for m in new:
            comp[m.id] = new
        head = new[0].id
        worst[head], why[head] = tier_w, reasons
        merged_ids.update(m.id for m in new)

    groups, done = [], set()
    for r in recs:
        c = comp[r.id]
        if len(c) > 1 and id(c) not in done:
            done.add(id(c))
            groups.append(_make_group(worst[c[0].id], c, why[c[0].id]))

    # Tier C: weak pairs between cases not already grouped
    for a, b in list(_town_pairs_c(recs)) + list(_placeless_pairs(recs)):
        if len(comp[a.id]) > 1 or len(comp[b.id]) > 1:
            continue
        e = _weak_edge(a, b)
        if e and _guards_ok(a, b):
            groups.append(_make_group("C", [a, b], [e[1]]))

    groups.sort(key=lambda g: (TIER_ORDER[g.tier], g.survivor_id, g.member_ids))
    return groups
