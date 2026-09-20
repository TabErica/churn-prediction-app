import json

import pandas as pd
from fake_logs import generate_event_logs

from churn.model import VOTING, ChurnModel
from churn.predict import main as predict_main
from churn.train import main as train_main


def test_train_then_predict_cli(tmp_path, capsys):
    train_logs = tmp_path / "train.parquet"
    generate_event_logs(n_users=150, seed=8).to_parquet(train_logs, index=False)
    model_path = tmp_path / "m.joblib"
    train_main(["--data", str(train_logs), "--folds", "2", "--out", str(model_path)])
    assert ChurnModel.load(model_path).model_name == VOTING
    assert json.loads(capsys.readouterr().out)["saved_to"] == str(model_path)

    test_logs = tmp_path / "test.parquet"
    generate_event_logs(n_users=40, seed=9).to_parquet(test_logs, index=False)
    template = tmp_path / "example_submission.csv"
    pd.DataFrame({"id": [*range(1, 41), 999], "target": 0}).to_csv(template, index=False)
    out = tmp_path / "submission.csv"

    predict_main(
        ["--model", str(model_path), "--data", str(test_logs), "--template", str(template),
         "--out", str(out)]
    )  # fmt: skip
    sub = pd.read_csv(out)
    assert sub["id"].tolist() == [*range(1, 41), 999]
    assert sub["target"].sum() == round(41 * 0.5)
    assert sub.loc[sub["id"] == 999, "target"].item() == 0
