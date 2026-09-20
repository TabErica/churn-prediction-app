import io

import pandas as pd
import pytest

from churn.data import REQUIRED_COLUMNS, basic_preprocess, filter_logs, load_logs, validate_schema

# ---------- importing ----------


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_load_logs_from_path(tmp_path, raw_logs, ext):
    path = tmp_path / f"logs.{ext}"
    getattr(raw_logs, f"to_{ext}")(path, index=False)
    loaded = load_logs(path)
    assert len(loaded) == len(raw_logs)
    assert set(loaded.columns) >= REQUIRED_COLUMNS


def test_load_logs_from_buffer_uses_file_name(raw_logs):
    buffer = io.BytesIO()
    raw_logs.to_parquet(buffer, index=False)
    buffer.seek(0)
    assert len(load_logs(buffer, file_name="logs.PARQUET")) == len(raw_logs)


def test_load_logs_reads_only_required_columns(tmp_path, raw_logs):
    path = tmp_path / "logs.parquet"
    raw_logs.assign(artist="x", userAgent="y").to_parquet(path, index=False)
    assert set(load_logs(path).columns) == REQUIRED_COLUMNS
    assert {"artist", "userAgent"} <= set(load_logs(path, all_columns=True).columns)


def test_load_parquet_missing_column_raises_value_error(tmp_path, raw_logs):
    path = tmp_path / "logs.parquet"
    raw_logs.drop(columns=["status"]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="status"):
        load_logs(path)


def test_load_logs_rejects_unknown_extension(tmp_path):
    path = tmp_path / "logs.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_logs(path)


def test_validate_schema_reports_missing_columns(raw_logs):
    with pytest.raises(ValueError, match="page"):
        validate_schema(raw_logs.drop(columns=["page"]))


def test_validate_schema_rejects_empty(raw_logs):
    with pytest.raises(ValueError, match="empty"):
        validate_schema(raw_logs.iloc[0:0])


# ---------- preprocessing ----------


def test_basic_preprocess_drops_anonymous_users(raw_logs):
    out = basic_preprocess(raw_logs)
    assert 0 not in set(out["userId"])
    assert len(out) == (raw_logs["userId"] != 0).sum()


def test_basic_preprocess_normalises_string_user_ids(raw_logs):
    """The real logs store userId as text; ids must still come out as integers."""
    as_text = raw_logs.assign(userId=raw_logs["userId"].astype(str))
    out = basic_preprocess(as_text)
    assert pd.api.types.is_integer_dtype(out["userId"])
    pd.testing.assert_frame_equal(out, basic_preprocess(raw_logs))


def test_basic_preprocess_parses_types_and_sorts(raw_logs):
    out = basic_preprocess(raw_logs)
    for col in ("ts", "registration", "time"):
        assert pd.api.types.is_datetime64_any_dtype(out[col])
    assert out["ts"].min() == pd.Timestamp("2018-10-01 10:00")
    assert out.equals(out.sort_values(["userId", "ts"]).reset_index(drop=True))


def test_basic_preprocess_strips_page_names(raw_logs):
    out = basic_preprocess(raw_logs)
    assert " NextSong " not in set(out["page"])
    assert (out["page"] == "NextSong").sum() == 4


def test_basic_preprocess_does_not_mutate_input(raw_logs):
    before = raw_logs.copy()
    basic_preprocess(raw_logs)
    pd.testing.assert_frame_equal(raw_logs, before)


# ---------- filtering ----------


@pytest.fixture
def logs(raw_logs):
    return basic_preprocess(raw_logs)


def test_filter_without_criteria_is_identity(logs):
    pd.testing.assert_frame_equal(filter_logs(logs), logs)


def test_filter_by_date_range_end_date_inclusive(logs):
    out = filter_logs(logs, start="2018-10-02", end="2018-10-03")
    assert out["ts"].min() >= pd.Timestamp("2018-10-02")
    assert out["ts"].max() < pd.Timestamp("2018-10-04")
    assert len(out) == 5  # 2 events on 10-02 (user 2) + 3 events on 10-03 (user 1)


def test_filter_by_end_timestamp_is_exact(logs):
    out = filter_logs(logs, end="2018-10-01 10:00")
    assert len(out) == 1


@pytest.mark.parametrize(
    ("kwargs", "expected_users"),
    [
        ({"levels": ["free"]}, {1}),
        ({"genders": ["F"]}, {2}),
        ({"pages": ["Error"]}, {2}),
        ({"levels": []}, set()),
    ],
)
def test_filter_by_categories(logs, kwargs, expected_users):
    assert set(filter_logs(logs, **kwargs)["userId"]) == expected_users
