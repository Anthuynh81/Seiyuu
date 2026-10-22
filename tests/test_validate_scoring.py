# ruff: noqa: E501 — a table of real sentences reads better unwrapped
"""Whisper-validation scoring on realistic transcripts (pure; no whisper model).

GOOD pairs are correct audio as whisper writes it (digits, abbreviations, respelled names); BAD
pairs are audio that must not ship (wrong numbers, dropped negations, dropped/looped/invented
clauses, many scattered errors). Collected from the 2026-10-22 3080 Ti benchmark and two rounds
of adversarial review of the scorer.
"""

import pytest

from seiyuu.normalize import normalize_text
from seiyuu.validate import Validator
from seiyuu.validate.scoring import score

# fmt: off
_PAD = " The wind moved slowly through the tall grass beside the old road, and nobody spoke for a while."
_LH = ("The lighthouse had not been lit in forty years, yet every evening at dusk the old keeper climbed its two hundred and twelve steps, wiped the salt from the great lens, and stood watching the horizon until the last of the fishing boats had found the harbor. Nobody in the village could remember who had first asked him to do it, and nobody had ever asked him to stop.")
_GOOD = [  # (source text (pre-normalize), whisper transcript)
 ("He had twelve hundred men." + _PAD, "He had 1,200 men." + _PAD),
 ("Fifteen hundred pilgrims came." + _PAD, "1,500 pilgrims came." + _PAD),
 ("twenty-five hundred men" + _PAD, "2,500 men" + _PAD),
 ("She dialed nine one one." + _PAD, "She dialed 911." + _PAD),
 ("Be there at seven fifteen sharp." + _PAD, "Be there at 7.15 sharp." + _PAD),
 ("In nineteen twelve, one thousand five hundred people died." + _PAD, "In 1912, 1500 people died." + _PAD),
 ("In nineteen twelve, fifteen hundred people died." + _PAD, "In 1912, 1,500 people died." + _PAD),
 ("It was June twenty-first, the longest day." + _PAD, "It was June 21, the longest day." + _PAD),
 ("on July fourth" + _PAD, "on July 4" + _PAD),
 ("On the third of May.", "On the 3 of May."),
 ("In nineteen eighty-four, it rained.", "In 1984, it rained."),
 ("It was 1905.", "It was 1905."),
 ("It was nineteen oh five.", "It was 1905."),
 ("in ten sixty-six", "in 1066"),
 ("She won the race at last.", "She one the race at last."),
 ("I didn't know.", "I did not know."),
 ("Mr. Darcy!", "Mr. Darcy!"),
 ("Then & now.", "Then and now."),
 ("Henry VIII ruled.", "Henry VIII ruled."),
 ("Her-MY-oh-nee GRAYN-jer laughed." + _PAD, "Hermione Granger laughed." + _PAD),
 ("Duh Nair Iss Tar Gair Ee En rode on." + _PAD, "Daenerys Targaryen rode on." + _PAD),
 ("oz-ee-MAN-dee-us was king." + _PAD, "Ozymandias was king." + _PAD),
 ("Her-my-oh-nee sighed." + _PAD, "Hermione sight." + _PAD),
 ("Ah'kalan dur'mahk ven'tari raised his blade." + _PAD, "Akalan Durmak Ventari raised his blade." + _PAD),
 ("Oh, Ah'kalan dur'mahk, you came back." + _PAD, "Akalan Durmak, you came back." + _PAD),
 ("He bought a hundred apples.", "He bought 100 apples."),
 ("They sang it twenty-four seven.", "They sang it 24/7."),
 ("It cost $2.50.", "It cost $2.50."),
 ("the one who sailed", "the ones who sailed"),
 ("the colour grey", "the color gray"),
 (_LH, _LH.replace("forty", "40").replace("two hundred and twelve", "212")),
 # third review round
 ("The city had 2.8 million people.", "The city had 2.8 million people."),
 ("It cost $1.5 million to build.", "It cost $1.5 million to build."),
 ("At one point twenty soldiers rushed the gate.", "At one point twenty soldiers rushed the gate."),
 ("At one point fifteen men stood at the gate.", "At one point, 15 men stood at the gate."),
 ("They walked two and a half miles to the farm.", "They walked 2.5 miles to the farm."),
 ("It took one and a half hours.", "It took 1.5 hours."),
 ('"One, two, three, jump!"', "1, 2, 3, jump!"),
 ("Five, four, three, two, one!", "5, 4, 3, 2, 1!"),
 ("The numbers were four, eight, fifteen, sixteen, twenty-three, forty-two.", "The numbers were 4, 8, 15, 16, 23, 42."),
 ("I dont know. I just dont think we should go down there, the boy said.", "I don't know. I just don't think we should go down there, the boy said."),
 ("I didn't see nothin', he said.", "I didn't see nothing, he said."),
 ("Nah, I'm fine, she said.", "No, I'm fine, she said."),
 ("Nevermind, he said.", "Never mind, he said."),
 ("I dunno, she said.", "I don't know, she said."),
 ("It's cold out, innit?", "It's cold out, isn't it?"),
 ("Ne'er had the hall been so quiet.", "Never had the hall been so quiet."),
 ('"No, no, no," she whispered, backing toward the door.', "No, no, she whispered, backing toward the door."),
 ("Mum said the neighbours had painted their grey fence a horrible colour, and she'd apologised twice already.", "Mom said the neighbors had painted their gray fence a horrible color, and she'd apologized twice already."),
 ('"P-P-Potter, w-what a s-surprise to s-see you here," said Quirrell.', "Potter, what a surprise to see you here, said Quirrell."),
 ('"N-n-no, I d-d-don\'t know," he stammered.', "No, I don't know, he stammered."),
 ("He had eaten half of it.", "He had eaten 1/2 of it."),
 ("The glass was three-quarters full.", "The glass was 3/4 full."),
 ("No one came to the door.", "Noone came to the door."),
 ("the one who sailed", "the ones who sailed"),
 ("She won the race at last.", "She one the race at last."),
 ("I went to the market.", "I went two the market."),
    # final review round
 ("Haha, no need to be so formal.", "No need to be so formal."),
 ("You really believed that? Hehe.", "You really believed that?"),
 ("Phew, that was close, she said.", "That was close, she said."),
 ("Meet me at 3 PM tomorrow by the old station.", "Meet me at 3pm tomorrow by the old station."),
 ("At 2:30 PM the bell rang in the hall.", "At 2:30pm the bell rang in the hall."),
 ("At one point two men came in from the yard.", "At one point, two men came in from the yard."),
 ("At one point, two of the guards fell asleep.", "At one point two of the guards fell asleep."),
 ("Wait 'til morning, she said.", "Wait until morning, she said."),
 ("He walked to the house at the end of the lane.", "He walked into the house at the end of the lane."),
]
_SWAPS = [("evening","morning"),("dusk","dust"),("climbed","climb"),("wiped","wipes"),("great","greet"),("watching","washing"),("last","least"),("fishing","finishing"),("found","fond"),("village","valley"),("remember","remain"),(" ever "," even ")]
_LH_12_SWAPS = _LH
for a, b in _SWAPS:
    _LH_12_SWAPS = _LH_12_SWAPS.replace(a, b, 1)
_BAD = [
 ("He climbed 212 steps.", "He climbed 213 steps."),
 ("He climbed 212 steps.", "He climbed 200 steps."),
 ("21 men", "20 men"),
 ("It cost $1,250.", "It cost $1,200."),
 ("105 years", "100 years"),
 ("It cost $3.50.", "It cost $3."),
 ("a hundred men", "two hundred men"),
 ("A thousand ships", "Ten thousand ships"),
 ("three sons", "the sons"),
 ("Nine of them", "None of them"),
 ("Ten soldiers", "Then soldiers"),
 ("the 1920s", "the 1930s"),
 ("in his twenties", "in his thirties"),
 ("Hundreds of people", "Thousands of people"),
 ("a dozen eggs", "two dozen eggs"),
 ("half of the gold", "a quarter of the gold"),
 ("It was 1847.", "It was 1874."),
 ("I did not kill him", "I did kill him"),
 (_LH, _LH.replace("had not been", "had been")),
 (_LH, "The lighthouse had not been lit in years, every evening the keeper climbed its steps, wiped the salt from the lens, and stood watching until the last boats had found the harbor. Nobody could remember who had first asked him, and nobody had asked him to stop."),
 (_LH, "The lighthouse had not been lit in forty years, oh the dark sea, yet every evening at dusk the old keeper with the grey beard climbed its two hundred and twelve steps, slowly, slowly, wiped the salt from the great lens, and stood watching the horizon. The bell rang out. until the last of the fishing boats had found the harbor. Nobody in the village could remember who had first asked him to do it, and nobody had ever asked him to stop. Thank you for listening."),
 (_LH, "The lighthouse had not been lit in forty years, yet every morning at noon the young sailor climbed its two hundred and twelve stairs, scraped the blood from the broken glass, and stood watching the horizon until the last of the fishing boats had left the harbor. Nobody in the city could remember who had first told him to do it, and nobody had ever told him to leave."),
 (_LH, _LH + " " + _LH[int(len(_LH)*0.7):]),
 (_LH, _LH[: int(len(_LH)*0.8)]),
 (_LH, _LH + " " + _LH),
 # third review round
 ("She had four children and a dog.", "She had eight children and a dog."),
 ("Two men came through the door.", "One men came through the door."),
 ("He had only one chance left.", "He had only two chance left."),
 ("He had two sons who fought in the war.", "He had sons who fought in the war."),
 ("I know who killed your father, and I can prove it.", "I know who killed your father, and I can."),
 ("He opened the door and saw that the room was empty.", "He opened the door and saw that the room"),
 ("He drew his sword and stepped into the darkness beyond the gate.", "He drew his sword and stepped into the darkness."),
 ("He left without his sword.", "He left with his sword."),
 ("The door was unlocked when she came home.", "The door was locked when she came home."),
 ("The pole was 3.5 metres long and painted red.", "The pole was three five metres long and painted red."),
 ("She could hardly breathe in the smoke.", "She could breathe in the smoke."),
    # final review round
 ("Harry said they would ride east at dawn, and that he would hate to lose even one more horse.", "Larry sad they could ride west at down, and that he would have to lost even one more horse."),
 (_LH, _LH_12_SWAPS),
 ("She had four sons and eight daughters.", "She had four eight sons and daughters."),
 ("The girls stared at their father in silence.", "The girls stared at their stared at their father in silence."),
]
# fmt: on


@pytest.mark.parametrize(("source", "heard"), _GOOD)
def test_correct_audio_passes(source: str, heard: str) -> None:
    verdict = score(normalize_text(source), heard)
    assert verdict.ok, (verdict.similarity, verdict.reason)
    assert verdict.reason is None


@pytest.mark.parametrize(("source", "heard"), _BAD)
def test_wrong_audio_fails_with_a_reason(source: str, heard: str) -> None:
    verdict = score(normalize_text(source), heard)
    assert not verdict.ok, verdict.similarity
    assert verdict.reason


def test_wrong_number_reason_names_both_numbers() -> None:
    verdict = score(normalize_text("He climbed 212 steps."), "He climbed 213 steps.")
    assert verdict.reason == (
        "wrong number: 'two hundred and twelve' heard as 'two hundred and thirteen'"
    )


def test_unscorable_transcript_fails_its_segment_instead_of_raising() -> None:
    class Whisper:
        def transcribe(self, path, **kwargs):
            seg = type("Seg", (), {"text": "1" * 400})()
            return [seg], None

    result = Validator(model=Whisper()).validate("seg.wav", "One.")
    assert not result.ok and result.score == 0.0
    assert result.reason.startswith("transcript could not be scored")


def test_lexicon_respelling_is_checked_against_the_plain_text() -> None:
    # The engine is given the respelling; whisper writes the real name.
    respelled = normalize_text(
        "Yoo-RIP-ih-deez looked up from the table, and for a long moment nobody in the room "
        "said anything."
    )
    plain = normalize_text(
        "Euripides looked up from the table, and for a long moment nobody in the room said "
        "anything."
    )
    heard = (
        "Euripides looked up from the table, and for a long moment nobody in a room said anything."
    )
    assert not score(respelled, heard).ok
    verdict = score(respelled, heard, alternates=(plain,))
    assert verdict.ok, verdict.reason
