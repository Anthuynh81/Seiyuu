"""Qwen-first casting, step 1: the cast plan (describer LLM + cast_plan.json + edits).

The LLM is a scripted fake (no Ollama): it answers per character from a table, keyed by the
character id it finds in the prompt. The cache, the plan file, the job runner and the routes
are real.
"""

import re
import time

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.attribute.cache import AttributionCache
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.local import OllamaPreflightError
from seiyuu.cli import main
from seiyuu.repository import JobState
from seiyuu.services import ServiceError
from seiyuu.services.cast_plan import (
    DEFAULT_NARRATOR_DESCRIPTION,
    NARRATOR_ROW,
    CastPlan,
    PlanRow,
    RowUpdate,
    apply_delivery,
    apply_updates,
    cached_sketcher,
    draft_plan,
    flatten_same_as,
    gather_evidence,
    is_first_person,
    load_plan,
    run_describe_cast,
    save_plan,
)
from seiyuu.voices.delivery import (
    DEFAULT_DIALOGUE,
    MAX_INSTRUCTION_CHARS,
    Delivery,
    DeliveryUpdate,
    load_delivery,
)
from seiyuu.voices.describer import CharacterEvidence, VoiceSketch, clean_sketch, sketch_character
from test_api_m6b1 import make_settings

BOOK = "ghost-00000000"
LONG = "I really did not expect to see you down here at this hour of the night, friend."


def _seg(i: int, kind: str, text: str, speaker: str | None = None) -> Segment:
    return Segment(block_id=f"ch001_b{i:04d}", type=kind, speaker=speaker, text=text)


def _report(first_person: bool = False) -> AttributionReport:
    """Kim: 6 lines (2 takes); Staff: 2 lines (1 take); New Employee: 7 lines; Ghost: none."""
    pron = "I" if first_person else "She"
    segs = [_seg(1, "narration", f"{pron} walked into the dark office and sat down at my desk.")]
    n = 2
    for k in range(6):
        segs.append(_seg(n, "dialogue", f"{LONG} ({k})" if k % 2 == 0 else "Yes.", "kim"))
        segs.append(_seg(n, "narration", f"Kim Soleum frowned. He looked tired, line {k}."))
        n += 1
    for k in range(2):
        segs.append(
            _seg(n, "dialogue", f"Welcome to Daydream, the staff said kindly ({k}).", "staff")
        )
        n += 1
    for k in range(7):
        segs.append(
            _seg(n, "dialogue", f"I need this job more than anything, honestly ({k}).", "new")
        )
        n += 1
    segs.append(
        _seg(n, "narration", f"{pron} was terrified, but I kept my face calm and my voice low.")
    )
    return AttributionReport(
        book_id=BOOK,
        provider_id="local",
        model_id="m",
        prompt_version="v1",
        registry=CharacterRegistry(
            characters=[
                Character(id="kim", canonical_name="Kim Soleum", gender="male"),
                Character(id="staff", canonical_name="Staff", description="A woman at the desk."),
                Character(id="new", canonical_name="New Employee"),
                Character(id="ghost", canonical_name="Ghost"),
            ]
        ),
        chapters=[AttributedChapter(index=1, title="One", segments=segs)],
    )


SKETCHES = {
    "kim": {"gender": "female", "age": "young adult", "voice_description": "A dry, calm young man.",
            "sample_line": 2, "same_as": None},
    "staff": {"gender": "female", "age": "adult", "voice_description": "A bright, polite woman.",
              "sample_line": 1, "same_as": None},
    "new": {"gender": "male", "age": "young adult", "voice_description": "A nervous young man.",
            "sample_line": 99, "same_as": "kim"},
}  # fmt: skip


class FakeLLM:
    provider_id = "local"
    model_id = "fake-describer"
    uses_gpu = False

    def __init__(self, replies=SKETCHES):
        self.replies = replies
        self.calls: list[str] = []

    def complete_structured(self, prompt, schema, *, tool_name="", tool_description=""):
        cid = re.search(r"^- id: (\S+)$", prompt, re.M).group(1)
        self.calls.append(cid)
        return dict(self.replies[cid])


def _fake_sketch(report):
    by_id = {ev.character_id: ev for ev in gather_evidence(report)}
    return lambda ev: clean_sketch(SKETCHES[ev.character_id], by_id[ev.character_id])


def _draft(report, **kw):
    return draft_plan(
        report,
        sketch=_fake_sketch(report),
        provider_id="local",
        model_id="m",
        prompt_version="v1",
        **kw,
    )


# -- evidence --------------------------------------------------------------------------------------


def test_evidence_is_busiest_first_and_skips_the_silent():
    evidence = gather_evidence(_report())
    assert [e.character_id for e in evidence] == ["new", "kim", "staff"]  # ghost has no lines
    kim = evidence[1]
    assert kim.line_count == 6
    # real lines first (the sample candidates), the short "Yes." after them as filler
    assert all(8 <= len(line.split()) <= 40 for line in kim.lines[:-1])
    assert kim.lines[-1] == "Yes."
    assert kim.mentions and all("Kim Soleum" in m for m in kim.mentions)
    assert ("kim", "Kim Soleum") not in kim.others and ("new", "New Employee") in kim.others


def test_point_of_view():
    assert is_first_person(_report(first_person=True))
    assert not is_first_person(_report(first_person=False))


# -- sketch cleaning ----------------------------------------------------------------------


def _ev(**kw):
    base = dict(character_id="a", name="A", lines=("one", "two"), others=(("b", "B"),))
    return CharacterEvidence(**{**base, **kw})


def test_clean_sketch_pins_the_reply_to_the_evidence():
    raw = {"gender": "male", "age": "adult", "voice_description": " ".join(["word"] * 50),
           "sample_line": 7, "same_as": "zzz"}  # fmt: skip
    s = clean_sketch(raw, _ev())
    assert s.sample_line == 1  # out of range -> the first line
    assert s.same_as is None  # not a listed character
    assert len(s.voice_description.split()) == 30
    assert clean_sketch({**raw, "same_as": "b", "sample_line": 2}, _ev()).same_as == "b"


def test_an_unusable_reply_is_a_value_error():
    class Bad(FakeLLM):
        def complete_structured(self, *a, **k):
            return {"gender": "robot", "age": "adult", "voice_description": "x", "sample_line": 1}

    with pytest.raises(ValueError, match="unusable reply"):
        sketch_character(Bad(), _ev(), prompt="- id: a")


# -- drafting -------------------------------------------------------------------------------------


def test_draft_builds_rows_takes_genders_and_the_narrator():
    plan = _draft(_report())
    rows = {r.character_id: r for r in plan.rows}
    assert list(rows) == ["new", "kim", "staff"]
    assert rows["kim"].gender == "male"  # the registry's gender wins over the model's "female"
    assert rows["new"].gender == "male"  # empty in the registry: the model fills it
    assert rows["kim"].takes == 2 and rows["staff"].takes == 1  # under 5 lines: 1 take
    assert rows["kim"].sample_index == 1  # the model's line 2, 0-based
    assert rows["new"].sample_index == 0  # line 99 doesn't exist
    assert rows["new"].same_as == "kim"
    assert rows["staff"].drafted_by == "llm"
    assert plan.narrator.character_id == NARRATOR_ROW
    assert plan.narrator.description == DEFAULT_NARRATOR_DESCRIPTION
    assert plan.narrator.sample_lines  # narration sentences to speak
    assert not plan.first_person and plan.protagonist_suggestion is None


def test_first_person_suggests_the_protagonist_through_same_as():
    plan = _draft(_report(first_person=True))
    assert plan.first_person
    assert plan.protagonist_suggestion == "kim"  # busiest is New Employee, the same person as Kim
    assert plan.narrator.same_as is None  # designed by default (decision Q1); only suggested


def test_a_bad_reply_leaves_a_note_and_a_fallback_row():
    report = _report()
    plan = draft_plan(
        report,
        sketch=lambda ev: None if ev.character_id == "staff" else _fake_sketch(report)(ev),
        provider_id="local", model_id="m", prompt_version="v1",
    )  # fmt: skip
    staff = next(r for r in plan.rows if r.character_id == "staff")
    assert staff.description == "A woman at the desk." and staff.drafted_by is None
    assert plan.notes == ["couldn't describe 'Staff': the model's reply was unusable"]


def test_a_redraft_keeps_edited_rows_and_carries_design_ids():
    plan = _draft(_report())
    edited = apply_updates(plan, [RowUpdate(character_id="staff", description="A tired clerk.")])
    edited.rows[1].design_id, edited.rows[1].voice_id = "0123456789ab", "kim_designed"
    again = _draft(_report(), existing=edited)
    rows = {r.character_id: r for r in again.rows}
    assert rows["staff"].description == "A tired clerk." and rows["staff"].edited
    assert rows["kim"].description == "A dry, calm young man."  # not edited: re-drafted
    assert (rows["kim"].design_id, rows["kim"].voice_id) == ("0123456789ab", "kim_designed")


def test_a_redraft_keeps_the_delivery_direction():
    # The render reads this block, and for a Breeze voice it is part of the cache key: a
    # "describe again" that reset it would silently re-record the book.
    plan = _draft(_report())
    assert plan.delivery == Delivery()  # a first draft starts from the defaults
    edited = apply_delivery(plan, DeliveryUpdate(enabled=False, dialogue="Quick and clipped."))
    assert not any(r.edited for r in edited.rows) and not edited.narrator.edited
    again = _draft(_report(), existing=edited)
    assert again.delivery == edited.delivery
    assert (again.delivery.enabled, again.delivery.dialogue) == (False, "Quick and clipped.")


def test_the_saved_plan_is_what_the_render_reads(tmp_path):
    plan = apply_delivery(_draft(_report()), DeliveryUpdate(inner_voice="  Hushed,  close. "))
    save_plan(tmp_path, plan)
    assert load_plan(tmp_path, BOOK) == plan
    assert load_delivery(tmp_path / BOOK) == plan.delivery
    assert plan.delivery.inner_voice == "Hushed, close."


def test_the_cache_answers_a_rerun_without_the_llm(tmp_path):
    report, llm = _report(), FakeLLM()
    with AttributionCache(tmp_path / "attribution.db") as cache:
        sketch = cached_sketcher(
            llm,
            cache,
            book_id=BOOK,
            prompts_dir=make_settings(tmp_path).prompts_dir,
            prompt_version="v1",
            first_person=False,
        )
        first = draft_plan(
            report, sketch=sketch, provider_id="local", model_id="m", prompt_version="v1"
        )
        assert sorted(llm.calls) == ["kim", "new", "staff"]
        second = draft_plan(
            report, sketch=sketch, provider_id="local", model_id="m", prompt_version="v1"
        )
    assert len(llm.calls) == 3  # all three came from the cache
    assert first.rows == second.rows


# -- edits --------------------------------------------------------------------------------


def test_updates_mark_rows_edited_and_validate():
    plan = _draft(_report())
    out = apply_updates(
        plan,
        [
            RowUpdate(character_id="staff", sample_index=1, gender="male", age="older adult"),
            RowUpdate(character_id="new", clear_same_as=True, keep_current=True),
            RowUpdate(character_id=NARRATOR_ROW, same_as="kim"),  # the narrator speaks as Kim
        ],
    )
    rows = {r.character_id: r for r in out.rows}
    assert (
        rows["staff"].sample_index == 1 and rows["staff"].gender == "male" and rows["staff"].edited
    )
    assert rows["new"].same_as is None and rows["new"].keep_current
    assert out.narrator.same_as == "kim" and out.narrator.edited
    assert not plan.rows[0].edited  # the input plan is untouched
    bad = [
        RowUpdate(character_id="nobody", description="abc"),
        RowUpdate(character_id="kim", same_as="kim"),
        RowUpdate(character_id="kim", same_as=NARRATOR_ROW),
        RowUpdate(character_id="kim", same_as="ghost"),  # not a row: no lines
        RowUpdate(character_id="kim", sample_index=40),
        RowUpdate(character_id="kim", gender="robot"),
        RowUpdate(character_id="kim", age="ancient"),
    ]
    for upd in bad:
        with pytest.raises(ServiceError):
            apply_updates(plan, [upd])


def test_same_as_chains_flatten_and_cycles_break():
    def row(cid, same=None):
        return PlanRow(character_id=cid, name=cid, description="x", same_as=same)

    rows = [row("a", "b"), row("b", "c"), row("c")]
    flatten_same_as(rows)
    assert [r.same_as for r in rows] == ["c", "c", None]
    loop = [row("a", "b"), row("b", "a"), row("c", "zzz")]
    flatten_same_as(loop)
    assert [r.same_as for r in loop] == [None, None, None]
    narrator = PlanRow(character_id=NARRATOR_ROW, name="N", description="x", same_as="a")
    rows = [row("a", "c"), row("c")]
    flatten_same_as(rows, narrator)
    assert narrator.same_as == "c"  # follows a's link to the voice that is designed


# -- the run, the routes, the CLI ---------------------------------------------------------


def _write(cfg, report):
    book_dir = cfg.books_dir / report.book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "attribution.json").write_text(report.model_dump_json(), encoding="utf-8")


def test_run_describe_cast_saves_the_plan(tmp_path):
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    messages: list[str] = []
    plan = run_describe_cast(cfg, BOOK, provider=FakeLLM(), progress=messages.append)
    assert load_plan(cfg.output_dir, BOOK) == plan
    assert plan.model_id == "fake-describer"
    assert "describing New Employee (1 of 3)" in messages


@pytest.fixture
def client(tmp_path, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("seiyuu.services.llm_advisory.build_provider", lambda *a, **k: llm)
    monkeypatch.setattr("seiyuu.api.routes.cast_plan.ollama_preflight", lambda *a, **k: None)
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.app = app
        yield c


def _wait(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.app.state.store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.02)
    pytest.fail("describe job never finished")


def test_api_describe_then_read_then_edit(client):
    assert client.get(f"/api/books/{BOOK}/cast-plan").status_code == 404  # not drafted yet
    resp = client.post(f"/api/books/{BOOK}/cast-plan/describe", json={})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["kind"] == "describe_cast" and job["book_id"] == BOOK
    assert _wait(client, job["job_id"]).state is JobState.SUCCEEDED

    plan = client.get(f"/api/books/{BOOK}/cast-plan").json()
    assert [r["character_id"] for r in plan["rows"]] == ["new", "kim", "staff"]
    assert plan["narrator"]["character_id"] == NARRATOR_ROW

    put = client.put(
        f"/api/books/{BOOK}/cast-plan",
        json={"updates": [{"character_id": "staff", "description": "A tired night clerk."}]},
    )
    assert put.status_code == 200, put.text
    staff = next(r for r in put.json()["rows"] if r["character_id"] == "staff")
    assert staff["description"] == "A tired night clerk." and staff["edited"]
    bad = client.put(
        f"/api/books/{BOOK}/cast-plan",
        json={"updates": [{"character_id": "kim", "same_as": "kim"}]},
    )
    assert bad.status_code == 422
    assert client.put(f"/api/books/{BOOK}/cast-plan", json={"updates": []}).status_code == 422


def test_api_edits_the_delivery_direction(client):
    url = f"/api/books/{BOOK}/cast-plan"
    job = client.post(f"{url}/describe", json={}).json()
    assert _wait(client, job["job_id"]).state is JobState.SUCCEEDED
    before = client.get(url).json()
    assert before["delivery"] == Delivery().model_dump()

    put = client.put(url, json={"delivery": {"enabled": False, "narration": " Clear,  steady. "}})
    assert put.status_code == 200, put.text
    after = put.json()
    assert after["delivery"]["enabled"] is False
    assert after["delivery"]["narration"] == "Clear, steady."
    assert after["delivery"]["dialogue"] == DEFAULT_DIALOGUE  # not named: unchanged
    assert after["rows"] == before["rows"]  # a delivery edit touches no row
    assert client.get(url).json()["delivery"] == after["delivery"]  # saved

    both = client.put(
        url,
        json={
            "updates": [{"character_id": "staff", "description": "A tired night clerk."}],
            "delivery": {"dialogue": ""},
        },
    )
    assert both.status_code == 200, both.text
    assert both.json()["delivery"]["dialogue"] == ""  # cleared: a plain clone for dialogue
    assert next(r for r in both.json()["rows"] if r["character_id"] == "staff")["edited"]

    assert client.put(url, json={"delivery": {}}).status_code == 422  # nothing to change
    assert client.put(url, json={}).status_code == 422
    assert client.put(url, json={"delivery": {"volume": 3}}).status_code == 422
    too_long = {"delivery": {"dialogue": "x" * (MAX_INSTRUCTION_CHARS + 1)}}
    assert client.put(url, json=too_long).status_code == 422
    assert client.get(url).json()["delivery"] == both.json()["delivery"]  # refusals saved nothing


def test_api_describe_refusals(client, monkeypatch):
    assert client.post("/api/books/nope/cast-plan/describe", json={}).status_code == 404

    def down(*a, **k):
        raise OllamaPreflightError("Ollama isn't running: start it", base_url="http://x")

    monkeypatch.setattr("seiyuu.api.routes.cast_plan.ollama_preflight", down)
    resp = client.post(f"/api/books/{BOOK}/cast-plan/describe", json={})
    assert resp.status_code == 503


def test_api_paid_describer_needs_confirmation(tmp_path, monkeypatch):
    cfg = make_settings(tmp_path, describe_provider="anthropic")
    _write(cfg, _report())
    with TestClient(create_app(settings=cfg)) as c:
        resp = c.post(f"/api/books/{BOOK}/cast-plan/describe", json={})
        assert resp.status_code == 402
        assert resp.json()["error"]["code"] == "payment_confirmation_required"
        no_key = c.post(f"/api/books/{BOOK}/cast-plan/describe", json={"confirm_paid": True})
        assert no_key.status_code == 503


def test_cli_describe_cast(tmp_path, monkeypatch):
    monkeypatch.setattr("seiyuu.services.llm_advisory.build_provider", lambda *a, **k: FakeLLM())
    for var, sub in (("BOOKS_DIR", "books"), ("OUTPUT_DIR", "output"), ("DATA_DIR", "data")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    from seiyuu.settings import get_settings

    get_settings.cache_clear()
    try:
        _write(get_settings(), _report(first_person=True))
        result = CliRunner().invoke(main, ["describe-cast", BOOK])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "3 characters, first person (fake-describer)" in result.output
    assert "same voice as: kim" in result.output
    assert "narrator could share the protagonist's voice: kim" in result.output
    assert isinstance(load_plan(tmp_path / "output", BOOK), CastPlan)


def test_voice_sketch_schema_is_closed():
    schema = VoiceSketch.model_json_schema()
    assert set(schema["properties"]["gender"]["enum"]) == {"male", "female", "unknown"}
    assert "elderly" in schema["properties"]["age"]["enum"]
    assert SegmentType.DIALOGUE  # (import used for clarity of the fixtures above)


def test_narrator_samples_skip_notices_and_front_matter():
    from seiyuu.services.cast_plan import narrator_samples

    report = _report()
    report.chapters[0].segments.insert(
        0,
        _seg(
            1,
            "narration",
            "※ Note : The company is not liable for any injuries that may occur during the "
            "course of the employee's duties here at all.",
        ),
    )
    samples = narrator_samples(report)
    assert samples and not any("※" in s for s in samples)
    assert all(s[0].isupper() for s in samples)


def test_a_character_with_many_good_lines_gets_no_filler():
    from seiyuu.services.cast_plan import MAX_LINES, _character_lines

    good = [f"This is a perfectly ordinary spoken line number {i} in the book." for i in range(20)]
    lines = _character_lines([*good, "Yes.", "No."])
    assert len(lines) == MAX_LINES and "Yes." not in lines  # no room left: no division by zero
    few = _character_lines([good[0], "Yes.", "No."])
    assert few == [good[0], "Yes.", "No."]


def test_wordless_lines_are_never_filler():
    from seiyuu.services.cast_plan import _character_lines

    assert _character_lines(["…!", "?!", "“…”", "Y-Yes, let’s!", "I"]) == ["Y-Yes, let’s!", "I"]
