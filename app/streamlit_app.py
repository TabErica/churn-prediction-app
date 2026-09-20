"""Streamlit front-end for the churn prediction project.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from churn.data import basic_preprocess, filter_logs, load_logs
from churn.features import build_features
from churn.labels import build_labels_and_cutoff
from churn.model import ChurnModel, make_submission

st.set_page_config(page_title="Churn Radar", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------- cached steps
# The real logs are several GB in memory. st.cache_resource keeps a single shared
# object instead of pickling a copy on every rerun (st.cache_data), and cached
# functions take a cheap string ``key`` instead of hashing huge DataFrames.
# None of the functions below mutate their inputs.
ROOT = Path(__file__).resolve().parents[1]
# The repository ships a 1,500-user extract; CHURN_DATA_DIR points at the full logs instead.
SAMPLE_DIR = ROOT / "data" / "sample"
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", SAMPLE_DIR)).resolve()
PRETRAINED_MODEL = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models" / "model.joblib"))
DEFAULT_TRAIN = DATA_DIR / "train.parquet"
DEFAULT_TEST = DATA_DIR / "test.parquet"


# Loaders return *cleaned* logs, so the raw frame is never kept in memory.
@st.cache_resource(
    show_spinner="Reading and cleaning logs (about a minute on the full log)…", max_entries=2
)
def read_local(path: str, mtime: float) -> pd.DataFrame:
    return basic_preprocess(load_logs(path))


@st.cache_resource(
    show_spinner="Engineering user features (a few minutes on the full log)…", max_entries=2
)
def user_features(_df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.Series]:
    y, cutoff = build_labels_and_cutoff(_df)
    feats, _ = build_features(_df, cutoff)
    return feats, y


@st.cache_resource(show_spinner="Scoring listeners…", max_entries=4)
def score(_model: ChurnModel, _logs: pd.DataFrame, key: str) -> pd.Series:
    return _model.predict_proba_from_logs(_logs, preprocessed=True)


@st.cache_resource(show_spinner="Loading the model…")
def read_model(path: str, mtime: float) -> ChurnModel:
    return ChurnModel.load(path)


def pick_logs(
    container, label: str, key: str, default_path: Path
) -> tuple[pd.DataFrame | None, str]:
    """Read a real log file from disk. Returns ``(clean_logs, cache_key)``."""
    path = container.text_input(label, str(default_path), key=f"{key}_path")
    if not Path(path).is_file():
        container.warning(f"File not found: {path}")
        return None, ""
    try:
        resolved, mtime = str(Path(path).resolve()), Path(path).stat().st_mtime
        return read_local(resolved, mtime), f"{resolved}|{mtime}"
    except Exception as err:  # corrupt file, wrong schema, unreadable path…
        container.error(f"Could not load the logs: {err}")
        return None, ""


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎧 Churn Radar")
st.sidebar.caption("Predict which listeners are about to cancel their subscription.")
logs, data_key = pick_logs(st.sidebar, "Training log (.parquet / .csv)", "train", DEFAULT_TRAIN)

if logs is None:
    st.info(
        f"No training data loaded yet. Put `train.parquet`, `test.parquet` and "
        f"`example_submission.csv` in `{DATA_DIR}`, or type another path in the sidebar."
    )
    st.stop()

source_name = Path(data_key.split("|")[0]).name

st.sidebar.subheader("Filters")
min_d, max_d = logs["ts"].min().date(), logs["ts"].max().date()
date_range = st.sidebar.date_input("Period", (min_d, max_d), min_value=min_d, max_value=max_d)
levels = st.sidebar.multiselect("Level", sorted(logs["level"].dropna().unique()))
genders = st.sidebar.multiselect("Gender", sorted(logs["gender"].dropna().unique()))
start, end = date_range if len(date_range) == 2 else (date_range[0], max_d)
view = filter_logs(  # None = no filter -> no copy of the (possibly huge) log
    logs,
    start=None if start == min_d else start,
    end=None if end == max_d else end,
    levels=levels or None,
    genders=genders or None,
)

if view.empty:
    st.warning("No events match these filters.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("Who is about to leave?")
sample_note = " · sample: 1,500 of 19,140 users" if DATA_DIR == SAMPLE_DIR else ""
st.caption(f"Source: {source_name}{sample_note} · {len(view):,} events after filters")

feats_all, y_all = user_features(logs, data_key)
users_in_view = view["userId"].unique()
y_view = y_all.reindex(users_in_view)
model = (
    read_model(str(PRETRAINED_MODEL), PRETRAINED_MODEL.stat().st_mtime)
    if PRETRAINED_MODEL.is_file()
    else None
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listeners", f"{len(users_in_view):,}")
c2.metric("Sessions", f"{view['sessionId'].nunique():,}")
c3.metric("Songs played", f"{(view['page'] == 'NextSong').sum():,}")
c4.metric("Churn rate", f"{y_view.mean():.1%}")

tab_explore, tab_features, tab_predict = st.tabs(["Explore", "Features", "Predict"])

# ---------------------------------------------------------------- explore
with tab_explore:
    daily = (
        view.groupby([view["ts"].dt.floor("D").rename("day"), "level"])
        .size()
        .rename("events")
        .reset_index()
    )
    st.subheader("Daily activity")
    st.altair_chart(
        alt.Chart(daily)
        .mark_area(opacity=0.7)
        .encode(x="day:T", y="events:Q", color="level:N", tooltip=["day:T", "level", "events"]),
    )

    left, right = st.columns(2)
    with left:
        st.subheader("What people do")
        pages = view["page"].value_counts().rename_axis("page").reset_index(name="events")
        st.altair_chart(
            alt.Chart(pages)
            .mark_bar()
            .encode(
                x=alt.X("events:Q", scale=alt.Scale(type="symlog")), y=alt.Y("page:N", sort="-x")
            )
        )
    with right:
        st.subheader("Churn by segment")
        seg = st.selectbox("Segment by", ["level", "gender"])
        seg_df = (
            feats_all.loc[users_in_view, [seg]]
            .join(y_all)
            .groupby(seg)["target"]
            .agg(churn_rate="mean", users="size")
            .reset_index()
        )
        st.altair_chart(
            alt.Chart(seg_df)
            .mark_bar()
            .encode(
                x=f"{seg}:N",
                y=alt.Y("churn_rate:Q", axis=alt.Axis(format="%")),
                tooltip=[seg, alt.Tooltip("churn_rate:Q", format=".1%"), "users"],
            )
        )

# ---------------------------------------------------------------- features
with tab_features:
    st.subheader("User-level features")
    st.caption(
        "Built only from events before each user's cutoff; "
        "'Cancel' pages are removed to avoid leakage."
    )
    table = feats_all.loc[users_in_view].join(y_all)
    st.dataframe(table, height=320)

    numeric = [c for c in table.select_dtypes("number").columns if c != "target"]
    feature = st.selectbox(
        "Compare churners vs. stayers on", numeric, index=numeric.index("n_events")
    )
    plot_df = table[[feature, "target"]].assign(
        group=lambda d: d["target"].map({0: "stayed", 1: "churned"})
    )
    st.altair_chart(
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max")
        .encode(x="group:N", y=f"{feature}:Q", color="group:N")
    )
    st.download_button(
        "Download features (CSV)", table.to_csv().encode(), "features.csv", "text/csv"
    )

# ---------------------------------------------------------------- predict
with tab_predict:
    st.subheader("Score new listeners")
    if model is None:
        st.error(f"No model at `{PRETRAINED_MODEL}`. Train one with `make train`.")
    else:
        m = model.metrics
        st.caption(
            f"{model.model_name}, trained on {m['n_users']:,} users · "
            f"out-of-fold ROC AUC {m['oof_auc']:.3f}"
        )
        test_logs, test_key = pick_logs(st, "Logs to score (.parquet / .csv)", "pred", DEFAULT_TEST)

        if test_logs is not None:
            proba = score(model, test_logs, test_key)
            churn_rate = float(y_all.mean())
            st.caption(
                f"Listeners are ranked by predicted risk, and the riskiest {churn_rate:.1%} are "
                f"flagged — the churn rate measured on the training log. At that cut-off the "
                f"model finds about two thirds of the leavers, and two thirds of the users it "
                f"flags do leave."
            )
            submission = make_submission(proba, pos_rate=churn_rate)
            result = (
                submission.join(proba, on="id")
                .sort_values("churn_proba", ascending=False, na_position="last")
                .set_index("id")
            )

            c1, c2 = st.columns(2)
            c1.metric("Users scored", f"{proba.notna().sum():,}")
            c2.metric("Flagged as churners", f"{submission['target'].sum():,}")
            st.altair_chart(
                alt.Chart(proba.rename("churn_proba").to_frame())
                .mark_bar()
                .encode(
                    alt.X(
                        "churn_proba:Q",
                        bin=alt.Bin(maxbins=30),
                        title="Predicted probability of cancelling",
                    ),
                    alt.Y("count()", title="Listeners"),
                )
            )
            st.dataframe(
                result,
                column_config={
                    "churn_proba": st.column_config.ProgressColumn(
                        "Risk of cancelling", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "target": st.column_config.CheckboxColumn("Flagged"),
                },
                height=320,
            )
            st.download_button(
                "Download scores (CSV)",
                submission.to_csv(index=False).encode(),
                "churn_scores.csv",
                "text/csv",
            )
