#!/usr/bin/env python3
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


# =========================
# User settings
# =========================
DATA_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA\Raw Data")
OUTPUT_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA\Week 1")

START_YYYYMM = "202401"

FILE_PATTERN = re.compile(r"^CRMLS(Listing|Sold)(\d{6})\.csv$", re.IGNORECASE)


# =========================
# Helper functions
# =========================
def most_recent_completed_month() -> str:
    today = datetime.today()
    year = today.year
    month = today.month
    if month == 1:
        return f"{year - 1}12"
    return f"{year}{month - 1:02d}"


def discover_files(data_dir: Path, start_yyyymm: str, end_yyyymm: str) -> pd.DataFrame:
    rows = []

    for path in data_dir.iterdir():
        if not path.is_file():
            continue

        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        dataset_type = match.group(1).capitalize()
        yyyymm = match.group(2)

        if start_yyyymm <= yyyymm <= end_yyyymm:
            rows.append({
                "file_name": path.name,
                "path": str(path),
                "dataset_type": dataset_type,
                "yyyymm": yyyymm,
            })

    return pd.DataFrame(rows).sort_values(["dataset_type", "yyyymm"]).reset_index(drop=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            continue

    raise ValueError(f"Cannot read file: {path}")


def combine_files(file_df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    subset = file_df[file_df["dataset_type"] == dataset_type]

    frames: List[pd.DataFrame] = []

    for row in subset.itertuples(index=False):
        print(f"Reading {row.file_name}...")
        df = safe_read_csv(row.path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# =========================
# Main
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_yyyymm = most_recent_completed_month()

    print(f"Combining data from {START_YYYYMM} to {end_yyyymm}")

    file_catalog = discover_files(DATA_DIR, START_YYYYMM, end_yyyymm)

    if file_catalog.empty:
        raise ValueError("No files found!")

    print("\nFiles found:")
    print(file_catalog)

    # =========================
    # Combine
    # =========================
    listing_combined = combine_files(file_catalog, "Listing")
    sold_combined = combine_files(file_catalog, "Sold")

    print("\nRow counts:")
    print(f"Listing: {len(listing_combined):,}")
    print(f"Sold: {len(sold_combined):,}")

    # =========================
    # Save
    # =========================
    listing_output = OUTPUT_DIR / f"CRMLSListingCombined_but_unfilter_with_property_type.csv"
    sold_output = OUTPUT_DIR / f"CRMLSSoldCombined_but_unfilter_with_property_type.csv"

    listing_combined.to_csv(listing_output, index=False, encoding="utf-8-sig")
    sold_combined.to_csv(sold_output, index=False, encoding="utf-8-sig")

    print("\nSaved files:")
    print(listing_output)
    print(sold_output)


if __name__ == "__main__":
    main()