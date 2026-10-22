"""Environment guards: dependency breakage that only shows up when a GPU engine loads.

These are cheap CPU imports, so the default suite (and CI) catches them long before a
multi-minute `pytest -m gpu` run does.
"""


def test_chatterbox_watermarker_imports() -> None:
    # resemble-perth swallows the ImportError and sets PerthImplicitWatermarker=None when
    # pkg_resources is missing (setuptools >= 82); Chatterbox then fails to construct with
    # "'NoneType' object is not callable". pyproject pins setuptools < 81 for this.
    import perth

    assert perth.PerthImplicitWatermarker is not None, (
        "perth's watermarker failed to import (is setuptools >= 82 installed? pkg_resources "
        "is gone); Chatterbox cannot load"
    )
