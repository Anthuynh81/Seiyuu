from seiyuu.settings import REPO_ROOT, Settings


def test_repo_root_points_at_project() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_directories_are_absolute() -> None:
    s = Settings(_env_file=None)
    assert s.books_dir.is_absolute()
    assert s.output_dir.is_absolute()
    assert s.voices_dir.is_absolute()


def test_cloud_keys_optional(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.anthropic_api_key is None
    assert s.elevenlabs_api_key is None
