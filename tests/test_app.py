"""Exercise the real workbench without invoking the language model."""

from pathlib import Path
import pytest


def test_workbench_search():
    if not Path("data/scifact.sqlite").exists() or not Path("models/encoder").exists():
        pytest.skip("Prepared local SciFact assets required")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=60
    ).run()
    assert not app.exception
    app.sidebar.selectbox[1].select("BM25").run()
    app.sidebar.toggle[0].set_value(False).run()
    app.text_area[0].set_value("Who is viewed as the first modern geologist?").run()
    app.button(key="search").click().run()
    assert not app.exception
    assert any(exp.label.startswith("[S1]") for exp in app.expander)

    app.sidebar.selectbox[1].select("Hybrid").run()
    app.sidebar.toggle[0].set_value(True).run()
    app.button(key="search").click().run()
    assert not app.exception
    assert any("James Hutton [S" in block.value for block in app.markdown)
    assert not app.error


def test_cautious_ui_and_collection_binding():
    if (
        not Path("reports/v2/agreement/answer-policy.json").exists()
        or not Path("models/verifier").exists()
    ):
        pytest.skip("Validated cautious assets required")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=60
    ).run()
    policy = next(
        item for item in app.sidebar.selectbox if item.label == "Answer policy"
    )
    policy.select("Cautious · validated").run()
    app.text_area[0].set_value(
        "What is the process of adding structure to real property or construction of buildings?"
    ).run()
    app.button(key="search").click().run()
    assert not app.exception and not app.error
    assert any("Building construction [S" in block.value for block in app.markdown)
    app.sidebar.selectbox[0].select("data/scifact.sqlite").run()
    app.button(key="search").click().run()
    assert app.error and "Policy does not match" in app.error[0].value
