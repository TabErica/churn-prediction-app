"""Smoke tests of the Streamlit UI with Streamlit's headless AppTest runner.

Each test points the app at a temporary folder holding fake logs of the same schema as
the real data, so the committed extract is never touched.
"""

from pathlib import Path

import pandas as pd
import pytest
from fake_logs import generate_event_logs
from streamlit.testing.v1 import AppTest

from churn.data import load_logs
from churn.model import train_from_logs

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Empty data folder and no pre-trained model (never touches the user's real data)."""
    monkeypatch.setenv("CHURN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "no_model.joblib"))
    return tmp_path


@pytest.fixture
def filled_data_dir(data_dir) -> Path:
    train = generate_event_logs(n_users=200, seed=3).assign(song="unused text column")
    train.to_parquet(data_dir / "train.parquet", index=False)
    generate_event_logs(n_users=60, seed=4).to_parquet(data_dir / "test.parquet", index=False)
    pd.DataFrame({"id": [*range(1, 61), 999], "target": 0}).to_csv(
        data_dir / "example_submission.csv", index=False
    )
    return data_dir


@pytest.fixture
def with_model(filled_data_dir, monkeypatch) -> Path:
    """Train a small model and put it where the app loads its pre-trained one from."""
    path = filled_data_dir / "model.joblib"
    logs = load_logs(filled_data_dir / "train.parquet")
    train_from_logs(logs, model_name="Logistic Regression", n_splits=2).save(path)
    monkeypatch.setenv("CHURN_MODEL_PATH", str(path))
    return path


def test_app_without_data_asks_for_it(data_dir):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "Who is about to leave?" not in [t.value for t in app.title]  # no dashboard yet
    assert "No training data loaded" in app.info[0].value
    assert any("File not found" in w.value for w in app.sidebar.warning)
    assert not app.get("file_uploader")  # no upload anywhere: real data is read from disk


def test_app_loads_local_data_by_default(filled_data_dir):
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Who is about to leave?"
    assert "train.parquet" in app.caption[0].value
    assert app.metric[0].value == "200"


def test_app_scores_with_the_pretrained_model(with_model):
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception
    assert any("out-of-fold ROC AUC" in c.value for c in app.caption)
    # Predict tab scores the local test.parquet and flags the riskiest listeners
    assert not app.slider  # the cut-off follows the observed churn rate, it is not a control
    assert any("riskiest" in c.value for c in app.caption)
    assert any(m.label == "Users scored" and m.value == "60" for m in app.metric)
    flagged = next(m for m in app.metric if m.label == "Flagged as churners")
    assert 0 < int(flagged.value) < 60
    assert len(app.dataframe) == 2  # features table + predictions table


def test_app_without_a_model_says_where_it_looked(filled_data_dir):
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception  # the other tabs still work
    assert any("No model at" in e.value for e in app.error)


def test_app_handles_bad_and_missing_local_files(data_dir):
    (data_dir / "train.parquet").write_bytes(b"")  # unreadable file
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception  # friendly error, not a crash
    assert any("Could not load the logs" in e.value for e in app.sidebar.error)
    app.sidebar.text_input(key="train_path").set_value(str(data_dir / "nope.parquet")).run()
    assert not app.exception
    assert any("File not found" in w.value for w in app.sidebar.warning)
