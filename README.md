# 🎧 Churn Radar — churn prediction for a music-streaming service

[![CI](https://github.com/TabErica/churn-prediction-app/actions/workflows/ci.yml/badge.svg)](https://github.com/TabErica/churn-prediction-app/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/badge/docker-ziqimeng0128%2Fchurn--radar-blue)](https://hub.docker.com/r/ziqimeng0128/churn-radar)

A Streamlit app, packaged in Docker, that predicts which listeners of a music-streaming service
are about to cancel their subscription — from their event logs alone (songs played, thumbs
up/down, adverts, errors, sessions…). The modelling was first done in a notebook; this
repository turns it into a tested Python package, an interactive app and a container image.

## Quick start

```bash
docker run --rm -p 8501:8501 ziqimeng0128/churn-radar:latest    # then open localhost:8501
```

The image ships a data extract and a trained model, so the dashboard works straight away. Or
locally, with Python 3.12:

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pinned deps + package in editable mode + pre-commit hooks
make run            # streamlit run app/streamlit_app.py
```

| Tab | Content |
| --- | --- |
| **Explore** | Daily activity, page usage, churn rate by subscription level or gender. Sidebar filters by period, level and gender. |
| **Features** | The ~70 user-level features; churners vs. stayers compared on any of them. CSV export. |
| **Predict** | Scores an event log, ranks listeners by risk, flags the riskiest share, exports the scores. |

Logs are read from disk, never uploaded, so a multi-GB file never goes through the browser.
The sidebar path box, `CHURN_DATA_DIR` and `CHURN_MODEL_PATH` point the app at other files.

## Method

1. **Cleaning** — drop anonymous users, parse timestamps, normalise page names.
2. **Labelling** — a user churns (`y = 1`) on reaching *Cancellation Confirmation*; the
   observation window stops there, or at their last event.
3. **Features** — ~70 per user, built only from events before that cutoff and with `Cancel`
   pages removed to prevent leakage: activity, recent 7-day activity, page counts and ratios,
   tenure, second-half vs. first-half trend, subscription changes, HTTP status, sessions,
   idle time.
4. **Imbalance** — the minority class is replicated `⌊n_major / n_minor⌋` times *inside each
   training fold only*, so validation folds stay untouched.
5. **Model** — soft vote of Logistic Regression and KNN,
   `(w_lr·p_lr + w_knn·p_knn) / (w_lr + w_knn)`, with integer weights 1–4 grid-searched on the
   out-of-fold predictions. Extra Trees and each member alone are available from the CLI.
6. **Decision** — fitting on oversampled data inflates the probabilities, so a fixed `p > 0.5`
   threshold would flag far too many people. Listeners are *ranked* instead and the riskiest
   share is flagged, which depends only on the ordering that ROC AUC measures.

5-fold out-of-fold ROC AUC, on the full training set (19,140 users) and on the committed
1,500-user extract:

| Model | Full | Extract |
| --- | --- | --- |
| **LR + KNN vote (4 : 1)** | **0.8953** | **0.8776** |
| Extra Trees | 0.9011 | — |
| Logistic Regression | 0.8945 | — |
| KNN (k = 50) | 0.8595 | — |

The weights are chosen on the same out-of-fold predictions that score them, so the ensemble AUC
is slightly optimistic; with 16 candidate pairs the bias is small.

The app flags the churn rate it measures on the training log. Out of fold that puts precision
and recall both near 66 %, within 0.005 F1 of the best cut-off available; flagging half the
users instead would reach 91 % recall but only 41 % precision.

## Data

The full logs come from a course competition (17.5 M train rows, ~740 MB) and **are not
redistributed** here. `data/sample/` ships a 27 MB extract instead, built to behave like the
real thing:

- **whole users, never whole rows** — 1,500 training and 750 test users keep *all* their
  events; sampling rows would destroy sessions, tenure and the churn labels;
- **stratified** on the label, reproducing the 22.3 % churn rate;
- **only the ten columns the pipeline reads** — `firstName`, `lastName`, `location` and
  `userAgent` are dropped, so no personal data is committed;
- drawn with `numpy.default_rng(42)`, so it can be rebuilt from the full logs.

`models/model.joblib` is the vote trained on that extract, loaded at start-up. Any folder with
the same three files works instead; `validate_schema` checks them on load.

## Command line

```bash
make train                            # LR + KNN voting on data/sample -> models/model.joblib
make submission                       # scores data/sample/test.parquet -> submission.csv
make train DATA=path/to/full/logs     # same targets on the full logs
python -m churn.train --model "Extra Trees" --folds 5
python -m churn.predict --data data/sample/test.parquet --pos-rate 0.3 --out sub.csv
```

Training is command-line only: the app scores, it does not fit.

## Repository layout

```
├── app/streamlit_app.py    # Streamlit UI, no business logic
├── src/churn/              # data.py, labels.py, features.py, model.py + train/predict CLIs
├── tests/                  # pytest, incl. fake_logs.py — schema-compatible logs for CI
├── data/sample/            # 1,500-user extract, committed
├── models/model.joblib     # trained on that extract, committed, loaded at start-up
├── .github/workflows/ci.yml
└── Dockerfile · Makefile · pyproject.toml · requirements*.txt · .pre-commit-config.yaml
```

## Tests and CI

```bash
make lint
make test           # 46 tests, ~99 % coverage
```

- **`test_data.py`** — importing parquet/CSV from a path or an in-memory buffer, unsupported
  formats, missing columns, empty files; preprocessing (anonymous users, text user ids, types,
  sorting, no input mutation); every filter, including the inclusive end day.
- **`test_features_and_model.py`** — labels and cutoffs, no post-cancellation leakage, column
  alignment, NaN/inf handling, oversampling row for row, the vote-weight search, submission
  alignment, a train → save → load → predict round trip, and reproducibility.
- **`test_cli.py`** / **`test_app.py`** — the CLI end to end; the app headless via `AppTest`
  (no-data screen, default load, scoring, missing model, unreadable files).

Tests run on fake logs from `tests/fake_logs.py`, never on the competition data. CI runs ruff,
then pytest on Python 3.12 behind a coverage gate of 85 %, then builds the image and
checks `/_stcore/health`. On `main` and `v*` tags it pushes to Docker Hub when
`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` are set.

## Reproducibility

- Runtime dependencies fully pinned (`requirements.txt` from a clean venv); Python version
  fixed by `.python-version` and the `python:3.12.11-slim` base image.
- Seeds fixed for CV splits, oversampling, Extra Trees and the fixtures; a test asserts that
  two training runs give the same AUC.
- All logic lives in an installable package that the app merely calls; ruff is enforced by
  pre-commit hooks and by CI.

## Notes

- Porting the notebook surfaced a bug: `basic_preprocess` now casts `userId` to an integer,
  because the raw logs store it as **text** while `example_submission.csv` uses integers, so
  `make_submission` matched nothing and returned a submission of all zeros. The fake logs used
  in testing had integer ids, which hid the mismatch — two tests now pin the behaviour on text
  ids.
- The app holds a large log once: only the ten needed columns are read, cleaned logs are cached
  with `st.cache_resource` keyed by path and modification time, and `filter_logs` returns the
  frame itself when no filter is active.
