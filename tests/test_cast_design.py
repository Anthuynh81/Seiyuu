"""Qwen-first casting, step 2: design the cast from the plan, assign it, swap takes.

Fakes only: the describer LLM (test_cast_plan.FakeLLM), the VoiceDesign engine and the speech
check (test_voice_design.Designer / Checker / Gpu). The plan, the design records, the voice
library, the assignment file, the job runner and the routes are real.
"""

import time

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.repository import JobState
from seiyuu.services import ServiceError
from seiyuu.services.assignment import load_existing_assignment, save_assignment
from seiyuu.services.cast_design import (
    FILLER_LINE,
    design_prompt,
    design_text,
    designing_rows,
    keep_other_take,
    run_design_cast,
)
from seiyuu.services.cast_plan import (
    NARRATOR_ROW,
    PlanRow,
    RowUpdate,
    apply_updates,
    load_plan,
    run_describe_cast,
    save_plan,
)
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceMeta
from test_api_m6b1 import make_settings
from test_cast_plan import BOOK, FakeLLM, _report, _write
from test_voice_design import Checker, Designer, Gpu


def _row(**kw):
    base = dict(character_id="a", name="A", description="A calm voice.")
    return PlanRow(**{**base, **kw})


# -- pure pieces ---------------------------------------------------------------------------------


def test_design_text_joins_short_lines_until_long_enough():
    lines = [
        "W-What are you…",
        "Get away from me right now.",
        "Please, I beg you, just leave me here and go.",
    ]
    row = _row(sample_lines=lines, sample_index=0)
    text = design_text(row)
    assert text.startswith("W-What are you…") and len(text.split()) >= 15
    one_long = _row(sample_lines=[" ".join(["word"] * 20), "next line here"], sample_index=0)
    assert design_text(one_long) == " ".join(["word"] * 20)  # already long enough
    wrap = _row(sample_lines=["one two three", "four five"], sample_index=1)
    # starts at the pick and wraps round; still under 15 words, so the neutral filler tops it up
    assert design_text(wrap) == f"four five one two three {FILLER_LINE}"
    with pytest.raises(ServiceError, match="no line"):
        design_text(_row(sample_lines=[]))


def test_design_text_never_runs_past_the_cap():
    row = _row(sample_lines=["short.", " ".join(["w"] * 45)], sample_index=0)
    # adding the 45-word line would make 46 words, so it is skipped (and the filler tops up)
    assert design_text(row) == f"short. {FILLER_LINE}"


def test_a_tiny_part_is_topped_up_to_a_usable_length():
    text = design_text(_row(sample_lines=["So that's why you pried open the wall?"]))
    assert text.startswith("So that's why") and len(text.split()) >= 15


def test_design_prompt_leads_with_age_and_gender():
    assert design_prompt(_row(age="young adult", gender="male")) == (
        "Young adult male voice. A calm voice."
    )
    assert design_prompt(_row(age="elderly")) == "Elderly voice. A calm voice."
    assert design_prompt(_row(gender="female")) == "Female voice. A calm voice."
    assert design_prompt(_row()) == "A calm voice."


def test_designing_rows_skip_shared_and_kept_voices_narrator_first(tmp_path):
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    plan = run_describe_cast(cfg, BOOK, provider=FakeLLM())  # new = same voice as kim
    assert [r.character_id for r in designing_rows(plan)] == [NARRATOR_ROW, "kim", "staff"]
    plan = apply_updates(
        plan,
        [
            RowUpdate(character_id="staff", keep_current=True),
            RowUpdate(character_id=NARRATOR_ROW, same_as="kim"),
        ],
    )
    assert [r.character_id for r in designing_rows(plan)] == ["kim"]


# -- the run --------------------------------------------------------------------------------------


@pytest.fixture
def described(tmp_path):
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    run_describe_cast(cfg, BOOK, provider=FakeLLM())
    return cfg


def test_design_cast_designs_keeps_and_assigns(described):
    cfg = described
    engine, gpu = Designer(), Gpu()
    messages: list[str] = []
    plan, assignment = run_design_cast(
        cfg, BOOK, engine=engine, validator=Checker(), gpu=gpu, progress=messages.append
    )
    rows = {r.character_id: r for r in plan.rows}
    # narrator 2 takes + kim 2 + staff 1 (under 5 lines); new shares kim's voice: none
    assert len(engine.designs) == 5
    kim_prompt = next(d for _t, d, _s in engine.designs if "dry, calm" in d)
    assert kim_prompt.startswith("Young adult male voice.")
    assert all(len(t.split()) >= 10 for t, _d, _s in engine.designs)
    assert gpu.freed == 1  # VoiceDesign held across rows, freed once at the end

    lib = VoiceLibrary(cfg.voices_dir)
    kim = lib.load(rows["kim"].voice_id)
    assert kim.kind is VoiceKind.CLONED and kim.engine == "qwen3tts"
    assert kim.tags == ["designed", BOOK]
    assert assignment.assignments["kim"] == rows["kim"].voice_id
    assert assignment.assignments["new"] == rows["kim"].voice_id  # same voice as Kim
    assert assignment.assignments["staff"] == rows["staff"].voice_id
    assert assignment.narrator_voice_id == plan.narrator.voice_id
    assert load_existing_assignment(cfg.output_dir, BOOK) == assignment
    assert load_plan(cfg.output_dir, BOOK) == plan
    assert "designing Kim Soleum (2 of 3)" in messages

    # resumable: nothing left to design, nothing changes
    again_engine = Designer()
    _plan2, again = run_design_cast(cfg, BOOK, engine=again_engine, validator=Checker(), gpu=Gpu())
    assert again_engine.designs == [] and again == assignment


def test_a_canceled_cast_keeps_finished_voices_and_resumes(described):
    cfg = described
    engine = Designer()
    calls = {"rows": 0}

    def cancel_on_second_row():
        calls["rows"] += 1
        if calls["rows"] == 4:  # row 1 check + its 2 takes, then row 2's check
            raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        run_design_cast(
            cfg,
            BOOK,
            engine=engine,
            validator=Checker(),
            gpu=Gpu(),
            check_cancel=cancel_on_second_row,
        )
    saved = load_plan(cfg.output_dir, BOOK)
    assert saved.narrator.voice_id is not None  # the first row finished and was saved
    assert load_existing_assignment(cfg.output_dir, BOOK) is None  # assigned only at the end

    rest = Designer()
    plan, assignment = run_design_cast(cfg, BOOK, engine=rest, validator=Checker(), gpu=Gpu())
    assert len(rest.designs) == 3  # kim 2 + staff 1; the narrator was not designed again
    assert assignment.narrator_voice_id == saved.narrator.voice_id


def test_keep_current_rows_and_a_shared_narrator(described):
    cfg = described
    lib = VoiceLibrary(cfg.voices_dir)
    lib.save(
        VoiceMeta(
            voice_id="my_pick",
            name="Mine",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_heart",
        )
    )
    save_assignment(
        VoiceAssignment(
            book_id=BOOK, narrator_voice_id="my_pick", assignments={"staff": "my_pick"}
        ),
        cfg.output_dir,
    )
    plan = load_plan(cfg.output_dir, BOOK)
    plan = apply_updates(
        plan,
        [
            RowUpdate(character_id="staff", keep_current=True),
            RowUpdate(character_id=NARRATOR_ROW, same_as="kim"),
        ],
    )
    save_plan(cfg.output_dir, plan)
    engine = Designer()
    plan, assignment = run_design_cast(cfg, BOOK, engine=engine, validator=Checker(), gpu=Gpu())
    kim_voice = next(r.voice_id for r in plan.rows if r.character_id == "kim")
    assert len(engine.designs) == 2  # only Kim
    assert assignment.assignments["staff"] == "my_pick"  # kept
    assert assignment.narrator_voice_id == kim_voice  # the narrator speaks as Kim


def test_editing_a_designed_row_redesigns_only_that_row(described):
    cfg = described
    run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())
    plan = load_plan(cfg.output_dir, BOOK)
    old_staff = next(r.voice_id for r in plan.rows if r.character_id == "staff")
    plan = apply_updates(plan, [RowUpdate(character_id="staff", description="A sleepy clerk.")])
    staff = next(r for r in plan.rows if r.character_id == "staff")
    assert staff.design_id is None and staff.voice_id is None
    save_plan(cfg.output_dir, plan)
    engine = Designer()
    plan, assignment = run_design_cast(cfg, BOOK, engine=engine, validator=Checker(), gpu=Gpu())
    assert [d for _t, d, _s in engine.designs] == ["Adult female voice. A sleepy clerk."]
    assert assignment.assignments["staff"] != old_staff


def test_swap_a_row_to_another_take(described):
    cfg = described
    plan, _ = run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())
    first = next(r.voice_id for r in plan.rows if r.character_id == "kim")
    plan, assignment = keep_other_take(cfg, BOOK, "kim", 2)
    second = next(r.voice_id for r in plan.rows if r.character_id == "kim")
    assert second != first
    assert assignment.assignments["kim"] == assignment.assignments["new"] == second
    # back to take 1 reuses the voice it already became
    plan, assignment = keep_other_take(cfg, BOOK, "kim", 1)
    assert assignment.assignments["kim"] == first
    with pytest.raises(ServiceError):
        keep_other_take(cfg, BOOK, "kim", 9)
    with pytest.raises(ServiceError, match="no designed voice"):
        keep_other_take(cfg, BOOK, "new", 1)  # shares Kim's voice; designs nothing


def test_design_cast_needs_a_plan(tmp_path):
    cfg = make_settings(tmp_path)
    with pytest.raises(ServiceError, match="describe the cast first"):
        run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())


# -- routes + CLI --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    designer = Designer()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda engine_id, **kw: designer)
    monkeypatch.setattr("seiyuu.validate.Validator", lambda **kw: Checker())
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    run_describe_cast(cfg, BOOK, provider=FakeLLM())
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.app = app
        yield c


def _wait(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.app.state.store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.02)
    pytest.fail("design-cast job never finished")


def test_api_design_cast_then_swap_a_take(client):
    resp = client.post(f"/api/books/{BOOK}/cast-plan/design")
    assert resp.status_code == 202, resp.text
    assert resp.json()["kind"] == "design_cast"
    job = _wait(client, resp.json()["job_id"])
    assert job.state is JobState.SUCCEEDED, job.error

    plan = client.get(f"/api/books/{BOOK}/cast-plan").json()
    kim = next(r for r in plan["rows"] if r["character_id"] == "kim")
    assert kim["voice_id"] and kim["design_id"]
    design = client.get(f"/api/designs/{kim['design_id']}").json()
    assert [t["take"] for t in design["takes"]] == [1, 2]  # hear the takes via #47's routes

    swap = client.post(f"/api/books/{BOOK}/cast-plan/rows/kim/take", json={"take": 2})
    assert swap.status_code == 200, swap.text
    body = swap.json()
    new_kim = next(r for r in body["plan"]["rows"] if r["character_id"] == "kim")["voice_id"]
    assert new_kim != kim["voice_id"]
    assert body["assignment"]["assignments"]["new"] == new_kim
    bad = client.post(f"/api/books/{BOOK}/cast-plan/rows/kim/take", json={"take": 7})
    assert bad.status_code == 422


def test_api_design_cast_needs_a_plan(tmp_path, monkeypatch):
    cfg = make_settings(tmp_path)
    _write(cfg, _report())
    with TestClient(create_app(settings=cfg)) as c:
        assert c.post(f"/api/books/{BOOK}/cast-plan/design").status_code == 404
        assert c.post("/api/books/nope/cast-plan/design").status_code == 404


def test_cli_design_cast(tmp_path, monkeypatch):
    monkeypatch.setattr("seiyuu.engines.get_engine", lambda engine_id, **kw: Designer())
    monkeypatch.setattr("seiyuu.validate.Validator", lambda **kw: Checker())
    dirs = {
        "BOOKS_DIR": "books",
        "OUTPUT_DIR": "output",
        "DATA_DIR": "data",
        "VOICES_DIR": "voices",
    }
    for var, sub in dirs.items():
        monkeypatch.setenv(var, str(tmp_path / sub))
    from seiyuu.settings import get_settings

    get_settings.cache_clear()
    try:
        cfg = get_settings()
        _write(cfg, _report())
        run_describe_cast(cfg, BOOK, provider=FakeLLM())
        result = CliRunner().invoke(main, ["design-cast", BOOK])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "designing 3 voices, 5 takes" in result.output
    assert "New Employee: shares kim's voice" in result.output
    assert "assignment saved" in result.output


def test_a_plan_edit_is_refused_while_the_cast_is_being_designed(client):
    """Design-cast loads the plan once and saves it after every row, so an edit accepted
    meanwhile would be written over; and the delivery block is what the render reads."""
    from seiyuu.repository import JobKind

    store = client.app.state.store
    job = store.create(book_id=BOOK, kind=JobKind.DESIGN_CAST, params={})
    url = f"/api/books/{BOOK}/cast-plan"
    before = client.get(url).json()
    for body in (
        {"delivery": {"enabled": False}},
        {"updates": [{"character_id": "staff", "description": "A tired night clerk."}]},
    ):
        resp = client.put(url, json=body)
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["code"] == "conflict"
    assert client.get(url).json() == before
    store.request_cancel(job.job_id)  # never enqueued: canceled on the spot
    assert client.put(url, json={"delivery": {"enabled": False}}).status_code == 200


def test_designed_voices_clone_on_breeze_where_it_is_set_up(described, monkeypatch):
    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)
    cfg = described
    plan, assignment = run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())
    lib = VoiceLibrary(cfg.voices_dir)
    designed = [r.voice_id for r in [plan.narrator, *plan.rows] if r.same_as is None]
    # Qwen designed the takes; the kept voices clone on Breeze
    assert {lib.load(v).engine for v in designed} == {"breeze"}
    assert assignment.narrator_voice_id == plan.narrator.voice_id
    kim = lib.load(next(r for r in plan.rows if r.character_id == "kim").voice_id)
    assert list(kim.settings) == ["breeze"] and kim.source == "qwen_design"


def test_redesign_moves_a_qwen_designed_cast_to_the_breeze_designer(described, monkeypatch):
    """A cast designed on Qwen is designed again by Breeze in one run; the old voices stay in the
    library, voices Breeze already made are kept, and a rerun has nothing left to do."""
    from test_voice_design import BreezeDesigner

    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)
    cfg = described
    lib = VoiceLibrary(cfg.voices_dir)
    old_plan, old_cast = run_design_cast(
        cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu()
    )
    old = {r.character_id: r.voice_id for r in [old_plan.narrator, *old_plan.rows]}
    assert {lib.load(v).source for v in old.values() if v} == {"qwen_design"}

    # without --redesign nothing is designed: every row already has a voice
    idle = BreezeDesigner()
    run_design_cast(cfg, BOOK, engine=idle, validator=Checker(), gpu=Gpu())
    assert idle.designs == []

    breeze = BreezeDesigner()
    plan, cast = run_design_cast(
        cfg, BOOK, engine=breeze, validator=Checker(), gpu=Gpu(), redesign=True
    )
    assert len(breeze.designs) == 5  # narrator 2 + kim 2 + staff 1, as the first time
    new = {r.character_id: r.voice_id for r in [plan.narrator, *plan.rows]}
    designed = [r for r in [plan.narrator, *plan.rows] if r.same_as is None]
    assert all(new[r.character_id] != old[r.character_id] for r in designed)
    assert {lib.load(r.voice_id).source for r in designed} == {"breeze_design"}
    assert {lib.load(r.voice_id).engine for r in designed} == {"breeze"}
    assert cast.narrator_voice_id == plan.narrator.voice_id != old_cast.narrator_voice_id
    assert all(lib.meta_path(v).is_file() for v in old.values() if v)  # nothing was deleted

    again = BreezeDesigner()
    run_design_cast(cfg, BOOK, engine=again, validator=Checker(), gpu=Gpu(), redesign=True)
    assert again.designs == []  # its own voices are kept: resumable, never a loop


def test_a_redesign_that_cannot_start_leaves_the_cast_linked(described, monkeypatch):
    """--redesign with the GPU held by another process designs nothing. It must not unlink the
    rows either: the cast keeps its voices AND its "hear takes" links, and a plain design-cast
    afterwards still has nothing to do."""
    from seiyuu.gpu import GpuBusyError
    from test_voice_design import BreezeDesigner

    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)
    cfg = described
    run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())
    before = load_plan(cfg.output_dir, BOOK)

    class Busy(Gpu):
        def acquire(self, engine, label):
            raise GpuBusyError("another seiyuu process holds the GPU")

    with pytest.raises(GpuBusyError):
        run_design_cast(
            cfg, BOOK, engine=BreezeDesigner(), validator=Checker(), gpu=Busy(), redesign=True
        )
    after = load_plan(cfg.output_dir, BOOK)
    linked = lambda p: [(r.character_id, r.voice_id) for r in [p.narrator, *p.rows]]  # noqa: E731
    # only the row whose turn had come let go of its old voice; the rest are untouched
    assert linked(after)[1:] == linked(before)[1:]
    assert [r.design_id for r in after.rows] == [r.design_id for r in before.rows]
    idle = Designer()
    run_design_cast(cfg, BOOK, engine=idle, validator=Checker(), gpu=Gpu())
    assert len(idle.designs) == 2  # just the one interrupted row (its 2 takes), not the cast


def test_a_cancelled_redesign_resumes_the_half_made_design(described, monkeypatch):
    from test_voice_design import BreezeDesigner

    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)
    cfg = described
    run_design_cast(cfg, BOOK, engine=Designer(), validator=Checker(), gpu=Gpu())
    calls = {"n": 0}

    def cancel_after_first_take():
        calls["n"] += 1
        if calls["n"] == 3:  # the row's check, take 1's check, then take 2's
            raise RuntimeError("canceled")

    first = BreezeDesigner()
    with pytest.raises(RuntimeError, match="canceled"):
        run_design_cast(
            cfg, BOOK, engine=first, validator=Checker(), gpu=Gpu(), redesign=True,
            check_cancel=cancel_after_first_take,
        )  # fmt: skip
    assert len(first.designs) == 1
    half = load_plan(cfg.output_dir, BOOK).narrator.design_id
    rest = BreezeDesigner()
    plan, _cast = run_design_cast(
        cfg, BOOK, engine=rest, validator=Checker(), gpu=Gpu(), redesign=True
    )
    assert plan.narrator.design_id == half  # the same design, finished
    assert len(rest.designs) == 4  # 5 takes in all, 1 already made
