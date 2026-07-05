"""Static, reviewed diminutive/nickname table feeding candidate generator G3.

Data only: a deterministic, curated map from a formal given name to its common
diminutives/nicknames. Precision over recall (SPEC): a missing entry silently drops a
candidate (no merge proposed), which is safe; a wrong entry could pair two distinct people,
so the table is hand-reviewed and small, and fuzzy/edit-distance matching stays OFF by
default. No I/O, no fuzzy logic — just a lookup other modules build a bidirectional index
from.

Entries are English/period-typical (the M2 fixtures are Austen); extend deliberately.
"""

from functools import lru_cache

# formal given name (casefolded) -> its accepted diminutives (casefolded).
_NICKNAMES: dict[str, frozenset[str]] = {
    "elizabeth": frozenset({"lizzy", "lizzie", "eliza", "beth", "bess", "betsy", "liza"}),
    "catherine": frozenset({"kitty", "kate", "katie", "cathy", "cat", "cass"}),
    "margaret": frozenset({"meg", "peggy", "maggie", "greta", "madge"}),
    "william": frozenset({"will", "bill", "billy", "willy", "liam"}),
    "richard": frozenset({"rich", "rick", "dick", "richie"}),
    "robert": frozenset({"rob", "bob", "bobby", "robbie", "bert"}),
    "charles": frozenset({"charlie", "chuck", "chas"}),
    "james": frozenset({"jim", "jimmy", "jamie"}),
    "john": frozenset({"jack", "johnny"}),
    "thomas": frozenset({"tom", "tommy"}),
    "edward": frozenset({"ed", "ned", "ted", "teddy", "eddie"}),
    "henry": frozenset({"harry", "hal", "hank"}),
    "anne": frozenset({"annie", "nan", "nancy"}),
    "mary": frozenset({"molly", "polly", "may"}),
    "jane": frozenset({"janey", "jenny"}),
    "susan": frozenset({"sue", "susie", "suky"}),
    "frances": frozenset({"fanny", "fran", "frankie"}),
    "eleanor": frozenset({"nell", "nelly", "ellie", "nora"}),
    "dorothy": frozenset({"dot", "dolly", "dora"}),
    "george": frozenset({"georgie"}),
}


@lru_cache
def _nickname_index() -> dict[str, frozenset[str]]:
    """Bidirectional index: each name (formal or diminutive) -> the set of names it links to.

    Built once and cached. A pair of given tokens is a nickname match iff one appears in the
    other's linked set. Diminutives that map to more than one formal name (rare but real)
    link to all of them; the LLM adjudicator still guards precision per pair.
    """
    index: dict[str, set[str]] = {}
    for formal, dims in _NICKNAMES.items():
        for dim in dims:
            index.setdefault(formal, set()).add(dim)
            index.setdefault(dim, set()).add(formal)
    return {name: frozenset(links) for name, links in index.items()}


def is_nickname_pair(given_a: str, given_b: str) -> bool:
    """True if the two given tokens are linked by the curated table (order-independent)."""
    a, b = given_a.casefold(), given_b.casefold()
    if a == b:
        return False
    return b in _nickname_index().get(a, frozenset())
