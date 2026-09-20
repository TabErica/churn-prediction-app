"""Data importing, validation, cleaning and filtering."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import IO

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "userId",
        "ts",
        "registration",
        "time",
        "page",
        "gender",
        "level",
        "length",
        "status",
        "sessionId",
    }
)


def load_logs(
    source: str | Path | IO[bytes],
    file_name: str | None = None,
    all_columns: bool = False,
) -> pd.DataFrame:
    """Load raw event logs from a parquet or CSV file.

    ``source`` may be a path or a file-like object (e.g. an in-memory buffer).
    For file-like objects the format is inferred from ``file_name``.
    Only ``REQUIRED_COLUMNS`` are read unless ``all_columns`` is True, which
    divides memory use by ~3 on the real logs (artist, song, userAgent… are unused).
    """
    name = str(file_name if file_name is not None else source).lower()
    if name.endswith(".parquet"):
        available = pq.ParquetFile(source).schema_arrow.names
        if hasattr(source, "seek"):
            source.seek(0)
        _check_columns(available)
        columns = None if all_columns else sorted(REQUIRED_COLUMNS)
        df = pd.read_parquet(source, columns=columns)
    elif name.endswith(".csv"):
        usecols = None if all_columns else (lambda c: c in REQUIRED_COLUMNS)
        df = pd.read_csv(source, usecols=usecols)
    else:
        raise ValueError(f"Unsupported file format for '{name}'. Use .parquet or .csv.")
    validate_schema(df)
    return df


def _check_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if mandatory columns are missing or the frame is empty."""
    _check_columns(df.columns)
    if df.empty:
        raise ValueError("The event log is empty.")


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw logs: drop anonymous users, parse timestamps, normalise pages, sort."""
    validate_schema(df)
    # The raw logs store userId as a string. Casting here keeps the anonymous-user filter
    # working and makes scores align with the integer ids of example_submission.csv.
    user_id = pd.to_numeric(df["userId"], errors="coerce")
    keep = user_id.notna() & (user_id != 0)
    df = df[keep].copy()  # single copy: the full log can be several GB
    df["userId"] = user_id[keep].astype("int64")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"])
    df["time"] = pd.to_datetime(df["time"])
    df["page"] = df["page"].astype(str).str.strip()
    return df.sort_values(["userId", "ts"]).reset_index(drop=True)


def filter_logs(
    df: pd.DataFrame,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    levels: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    pages: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter preprocessed logs. ``None`` means "no filter" for that criterion.

    ``start`` is inclusive. ``end`` is inclusive; a date without a time
    component includes the whole day. When no criterion is given the input
    frame itself is returned (no copy), which matters for multi-GB logs.
    """
    if all(v is None for v in (start, end, levels, genders, pages)):
        return df
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["ts"] >= pd.Timestamp(start)
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():
            mask &= df["ts"] < end_ts + pd.Timedelta(days=1)
        else:
            mask &= df["ts"] <= end_ts
    if levels is not None:
        mask &= df["level"].isin(list(levels))
    if genders is not None:
        mask &= df["gender"].isin(list(genders))
    if pages is not None:
        mask &= df["page"].isin(list(pages))
    return df[mask].copy()
