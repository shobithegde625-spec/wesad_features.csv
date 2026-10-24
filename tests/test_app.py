"""Streamlit checks use the real locally trained ECG bundle when available."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = any(p.with_name("metadata.json").exists() for p in ROOT.glob("artifacts/*/model.joblib"))


@pytest.mark.skipif(not AVAILABLE, reason="Run train_model.py first to exercise the dashboard")
def test_dashboard_pages_and_prediction():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["Prediction", "History", "Analysis", "Performance", "Monitoring", "Overview"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page == "Prediction":
            app.button[0].click().run()
            assert not app.exception
            assert len(app.session_state["history"]) == 1
    app.sidebar.radio[0].set_value("Monitoring").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["cursor"] > 0
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["running"] is False
    # Automatic completion must refresh controls so the recording can be restarted.
    # The exact participant window count varies; drive completion deterministically.
    from pulsepoint.data import load_dataset
    frame = load_dataset(Path(app.session_state["context"][2]), min_subjects=1)
    chosen_subject = app.session_state["context"][5]
    app.session_state["cursor"] = len(frame[frame.subject == chosen_subject])-1
    app.session_state["running"] = True
    app.run()
    assert not app.exception
    assert app.session_state["running"] is False
    assert app.button[0].disabled is False
