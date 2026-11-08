"""Deterministic gender back-fill — a once-per-book post-pass run AFTER alias resolution.

Small attribution models often leave ``Character.gender`` empty (15 of 24 characters in one
web novel, the protagonist among them), and casting reads an empty gender as female. This pass
fills ONLY empty genders, from evidence the book itself states, and never overwrites a gender
the model gave. Precision over recall: when the evidence is thin or mixed the gender stays
unknown (casting's existing warning then tells the user), because a confidently wrong voice is
worse than a flagged one.

Three sources, in order:

1. **Title.** A gendered honorific in the canonical name (``Mr. Darcy``, ``Mrs. Bennet``,
   ``Miss Lucas``) — the same table ``aliases._conflict`` already trusts. A child title
   (``Master``) is a generation signal, not a gender one, so it is not used.
2. **A gendered noun in the name itself**, for descriptive names: ``pointy hat woman``,
   ``Audrey Hall's maidservant``, ``the king's daughter`` (a possessive word names someone
   else, so it is ignored). Every gendered noun left must agree.
3. **Pronouns after the name.** In narration, a sentence that names exactly ONE character,
   followed by a gendered pronoun later in that sentence or in the next one (when the next
   sentence names no character): ``Kwak Jaekang nodded. He ...``. Filled only with at least
   ``min_votes`` votes and a ``min_share`` majority. Measured on three local books, every
   filled gender agreed with the genders the model DID give (12 of 12).

A name is only matched as a proper name: keys shorter than 3 characters, lowercase keys, and
single-word keys the book also uses as an ordinary lowercase word (``host``, ``staff``,
``ghost``) are skipped, so a role-named character isn't credited with every "the host ... he".

Pure: no I/O, no LLM. Mutates the registry in place (as ``resolve_registry_aliases`` does)
and returns review notes naming each fill and its evidence.
"""

import re
from collections import Counter

from seiyuu.attribute.aliases import _TITLE_GENDER, _title_class
from seiyuu.attribute.models import AttributedChapter, Character, CharacterRegistry, SegmentType

MIN_VOTES = 3
MIN_SHARE = 0.8

_MALE_PRONOUNS = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
_FEMALE_PRONOUNS = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_WORD = re.compile(r"[a-z]+")

# Whole words that state a person's gender when they appear in that person's own (descriptive)
# name. Deliberately narrow: role words used for both genders (priest, doctor, host) are absent.
_NOUN_GENDER = {
    **dict.fromkeys(
        "man men boy guy gentleman father dad son brother husband uncle nephew grandfather "
        "grandpa king prince duke butler waiter landlord mister".split(),
        "male",
    ),
    **dict.fromkeys(
        "woman women girl lady gal mother mom mum daughter sister wife aunt niece grandmother "
        "grandma queen princess duchess madam maid maidservant nun waitress landlady actress "
        "hostess".split(),
        "female",
    ),
}
_POSSESSIVE = re.compile(r"[^\W\d_]+['’]s\b")
_NAME_WORD = re.compile(r"[^\W\d_]+")

# The narrator record speaks the prose itself; pronouns near "Narrator" say nothing about it.
_SKIP_IDS = frozenset({"narrator"})


def _name_keys(char: Character, common_words: set[str]) -> list[str]:
    keys = []
    for name in {char.canonical_name, *char.aliases}:
        name = name.strip()
        if len(name) < 3 or not name[0].isupper():
            continue
        if " " not in name and name.casefold() in common_words:
            continue
        keys.append(name)
    return sorted(keys, key=len, reverse=True)  # longest first: "Kim Soleum" before "Kim"


def _noun_gender(name: str) -> str | None:
    """The gender a descriptive name states about its own subject, or None (absent/mixed)."""
    own = _POSSESSIVE.sub(" ", name)  # "the king's daughter" -> "the   daughter"
    found = {_NOUN_GENDER[w] for w in _NAME_WORD.findall(own.casefold()) if w in _NOUN_GENDER}
    return found.pop() if len(found) == 1 else None


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _vote(text: str) -> str | None:
    male, female = len(_MALE_PRONOUNS.findall(text)), len(_FEMALE_PRONOUNS.findall(text))
    if male and not female:
        return "male"
    if female and not male:
        return "female"
    return None


def count_pronoun_votes(
    registry: CharacterRegistry, chapters: list[AttributedChapter]
) -> dict[str, Counter[str]]:
    """Per character id, the male/female pronoun votes that follow a lone mention of its name."""
    narration = [
        seg.text
        for chapter in chapters
        for seg in chapter.segments
        if seg.type is SegmentType.NARRATION
    ]
    common_words = {w for text in narration for w in _WORD.findall(text)}
    patterns: dict[str, re.Pattern[str]] = {}
    for char in registry.characters:
        if char.id in _SKIP_IDS:
            continue
        keys = _name_keys(char, common_words)
        if keys:
            alternatives = "|".join(map(re.escape, keys))
            patterns[char.id] = re.compile(rf"\b(?:{alternatives})\b")

    votes: dict[str, Counter[str]] = {cid: Counter() for cid in patterns}
    for text in narration:
        sentences = _sentences(text)
        for i, sentence in enumerate(sentences):
            hits = [(cid, m) for cid, p in patterns.items() if (m := p.search(sentence))]
            if len(hits) != 1:
                continue  # no name, or two characters: whose pronoun is it?
            cid, match = hits[0]
            following = sentence[match.end() :]
            if i + 1 < len(sentences):
                nxt = sentences[i + 1]
                if not any(p.search(nxt) for p in patterns.values()):
                    following = f"{following} {nxt}"
            verdict = _vote(following)
            if verdict:
                votes[cid][verdict] += 1
    return votes


def infer_genders(
    registry: CharacterRegistry,
    chapters: list[AttributedChapter],
    *,
    min_votes: int = MIN_VOTES,
    min_share: float = MIN_SHARE,
) -> list[str]:
    """Fill empty genders: title, name noun, then pronouns. Mutates ``registry``, returns notes."""
    notes: list[str] = []
    unknown = [
        c
        for c in registry.characters
        if c.id not in _SKIP_IDS and not (c.gender or "").strip()  # never overwrite a given one
    ]
    if not unknown:
        return notes

    votes: dict[str, Counter[str]] | None = None
    for char in unknown:
        title = _TITLE_GENDER.get(_title_class(char.canonical_name))
        if title:
            char.gender = title
            notes.append(
                f"gender inferred for {char.canonical_name!r}: {title} "
                f"(from the title {char.canonical_name.split()[0]!r})"
            )
            continue
        noun = _noun_gender(char.canonical_name)
        if noun:
            char.gender = noun
            notes.append(
                f"gender inferred for {char.canonical_name!r}: {noun} (from the name itself)"
            )
            continue
        if votes is None:
            votes = count_pronoun_votes(registry, chapters)
        tally = votes.get(char.id, Counter())
        total = tally["male"] + tally["female"]
        if total < min_votes:
            continue
        gender, count = tally.most_common(1)[0]
        if count / total < min_share:
            continue
        char.gender = gender
        notes.append(
            f"gender inferred for {char.canonical_name!r}: {gender} "
            f"({count} of {total} pronouns after the name)"
        )
    return notes
