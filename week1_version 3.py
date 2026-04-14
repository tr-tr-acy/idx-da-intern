#!/usr/bin/env python3
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# =========================
# User settings
# =========================
DATA_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA\Raw Data")
OUTPUT_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA\Week 1\Week 1_version 3")

START_YYYYMM = "202401"

# Expected file names:
#   CRMLSListingYYYYMM.csv
#   CRMLSSoldYYYYMM.csv
FILE_PATTERN = re.compile(r"^CRMLS(Listing|Sold)(\d{6})\.csv$", re.IGNORECASE)


def most_recent_completed_month() -> str:
    today = datetime.today()
    year = today.year
    month = today.month
    if month == 1:
        return f"{year - 1}12"
    return f"{year}{month - 1:02d}"


def discover_files(data_dir: Path, start_yyyymm: str, end_yyyymm: str) -> pd.DataFrame:
    rows = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    for path in data_dir.iterdir():
        if not path.is_file():
            continue

        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        dataset_type = match.group(1).capitalize()
        yyyymm = match.group(2)

        if start_yyyymm <= yyyymm <= end_yyyymm:
            rows.append(
                {
                    "file_name": path.name,
                    "path": str(path),
                    "dataset_type": dataset_type,
                    "yyyymm": yyyymm,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["file_name", "path", "dataset_type", "yyyymm"])

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset_type", "yyyymm", "file_name"])
        .reset_index(drop=True)
    )


def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e

    raise ValueError(f"Unable to read CSV: {path}\nLast error: {last_error}")


def combine_monthly_files(file_df: pd.DataFrame, dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = (
        file_df[file_df["dataset_type"].str.lower() == dataset_type.lower()]
        .sort_values("yyyymm")
        .reset_index(drop=True)
    )

    if subset.empty:
        raise FileNotFoundError(f"No {dataset_type} files found in requested range.")

    frames: List[pd.DataFrame] = []
    file_count_rows = []

    for row in subset.itertuples(index=False):
        df = safe_read_csv(row.path)
        frames.append(df)

        file_count_rows.append(
            {
                "dataset_type": dataset_type,
                "file_name": row.file_name,
                "yyyymm": row.yyyymm,
                "row_count": len(df),
                "column_count": df.shape[1],
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    counts_df = pd.DataFrame(file_count_rows)

    return combined, counts_df


def residential_filter(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "PropertyType" not in df.columns:
        raise KeyError(
            f"'PropertyType' column not found in {dataset_name} dataset. "
            f"Available columns sample: {list(df.columns[:20])}"
        )

    prop = df["PropertyType"].astype(str).str.strip().str.lower()
    return df[prop == "residential"].copy()


def build_summary_log(
    listing_before_concat: int,
    listing_after_concat: int,
    sold_before_concat: int,
    sold_after_concat: int,
    listing_before_filter: int,
    listing_after_filter: int,
    sold_before_filter: int,
    sold_after_filter: int,
) -> pd.DataFrame:
    rows = [
        {"dataset_type": "Listing", "step": "sum_of_monthly_file_rows_before_concat", "row_count": listing_before_concat},
        {"dataset_type": "Listing", "step": "row_count_after_concat", "row_count": listing_after_concat},
        {"dataset_type": "Listing", "step": "row_count_before_residential_filter", "row_count": listing_before_filter},
        {"dataset_type": "Listing", "step": "row_count_after_residential_filter", "row_count": listing_after_filter},
        {"dataset_type": "Sold", "step": "sum_of_monthly_file_rows_before_concat", "row_count": sold_before_concat},
        {"dataset_type": "Sold", "step": "row_count_after_concat", "row_count": sold_after_concat},
        {"dataset_type": "Sold", "step": "row_count_before_residential_filter", "row_count": sold_before_filter},
        {"dataset_type": "Sold", "step": "row_count_after_residential_filter", "row_count": sold_after_filter},
    ]
    return pd.DataFrame(rows)


def write_text_log(
    log_path: Path,
    start_yyyymm: str,
    end_yyyymm: str,
    listing_file_counts: pd.DataFrame,
    sold_file_counts: pd.DataFrame,
    listing_before_concat: int,
    listing_after_concat: int,
    sold_before_concat: int,
    sold_after_concat: int,
    listing_before_filter: int,
    listing_after_filter: int,
    sold_before_filter: int,
    sold_after_filter: int,
) -> None:
    lines = []
    lines.append("IDX Exchange - Week 1 Monthly Dataset Aggregation Log")
    lines.append("=" * 70)
    lines.append(f"Date run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data range: {start_yyyymm} to {end_yyyymm}")
    lines.append(f"Raw data folder: {DATA_DIR}")
    lines.append(f"Output folder: {OUTPUT_DIR}")
    lines.append("")

    lines.append("Week 1 requirement:")
    lines.append("Concatenate all monthly Listing and Sold files from January 2024")
    lines.append("through the most recently completed calendar month, filter to")
    lines.append("PropertyType == 'Residential', and save as new CSVs.")
    lines.append("")

    lines.append("LISTING FILES INCLUDED")
    lines.append("-" * 70)
    for row in listing_file_counts.itertuples(index=False):
        lines.append(
            f"{row.file_name} | yyyymm={row.yyyymm} | rows={row.row_count:,} | cols={row.column_count:,}"
        )
    lines.append("")

    lines.append("SOLD FILES INCLUDED")
    lines.append("-" * 70)
    for row in sold_file_counts.itertuples(index=False):
        lines.append(
            f"{row.file_name} | yyyymm={row.yyyymm} | rows={row.row_count:,} | cols={row.column_count:,}"
        )
    lines.append("")

    lines.append("ROW COUNT SUMMARY")
    lines.append("-" * 70)
    lines.append("Listing dataset:")
    lines.append(f"  Sum of monthly file rows before concat: {listing_before_concat:,}")
    lines.append(f"  Row count after concat:                {listing_after_concat:,}")
    lines.append(f"  Row count before Residential filter:   {listing_before_filter:,}")
    lines.append(f"  Row count after Residential filter:    {listing_after_filter:,}")
    lines.append("")
    lines.append("Sold dataset:")
    lines.append(f"  Sum of monthly file rows before concat: {sold_before_concat:,}")
    lines.append(f"  Row count after concat:                 {sold_after_concat:,}")
    lines.append(f"  Row count before Residential filter:    {sold_before_filter:,}")
    lines.append(f"  Row count after Residential filter:     {sold_after_filter:,}")
    lines.append("")

    lines.append("NOTES")
    lines.append("-" * 70)
    lines.append("1. 'Before concat' = sum of all included monthly file row counts.")
    lines.append("2. 'After concat' should match the summed row counts if no rows were lost.")
    lines.append("3. Residential filter applied using PropertyType == 'Residential' (case-insensitive after trimming).")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_yyyymm = most_recent_completed_month()
    print(f"Using file range: {START_YYYYMM} to {end_yyyymm}")

    file_catalog = discover_files(DATA_DIR, START_YYYYMM, end_yyyymm)
    if file_catalog.empty:
        raise FileNotFoundError(
            f"No matching files found in {DATA_DIR} for range {START_YYYYMM} to {end_yyyymm}"
        )

    print("\nDiscovered files:")
    print(file_catalog.to_string(index=False))

    listings_all, listing_file_counts = combine_monthly_files(file_catalog, "Listing")
    sold_all, sold_file_counts = combine_monthly_files(file_catalog, "Sold")

    listing_before_concat = int(listing_file_counts["row_count"].sum())
    sold_before_concat = int(sold_file_counts["row_count"].sum())

    listing_after_concat = len(listings_all)
    sold_after_concat = len(sold_all)

    listing_before_filter = len(listings_all)
    sold_before_filter = len(sold_all)

    listings_residential = residential_filter(listings_all, "Listing")
    sold_residential = residential_filter(sold_all, "Sold")

    listing_after_filter = len(listings_residential)
    sold_after_filter = len(sold_residential)

    listing_output = OUTPUT_DIR / f"CRMLSListingCombined_Residential_{START_YYYYMM}_to_{end_yyyymm}.csv"
    sold_output = OUTPUT_DIR / f"CRMLSSoldCombined_Residential_{START_YYYYMM}_to_{end_yyyymm}.csv"
    file_catalog_output = OUTPUT_DIR / "week1_file_catalog.csv"
    listing_counts_output = OUTPUT_DIR / "week1_listing_file_row_counts.csv"
    sold_counts_output = OUTPUT_DIR / "week1_sold_file_row_counts.csv"
    summary_output = OUTPUT_DIR / "week1_row_count_summary.csv"
    txt_log_output = OUTPUT_DIR / "week1_aggregation_log.txt"

    listings_residential.to_csv(listing_output, index=False, encoding="utf-8-sig")
    sold_residential.to_csv(sold_output, index=False, encoding="utf-8-sig")
    file_catalog.to_csv(file_catalog_output, index=False, encoding="utf-8-sig")
    listing_file_counts.to_csv(listing_counts_output, index=False, encoding="utf-8-sig")
    sold_file_counts.to_csv(sold_counts_output, index=False, encoding="utf-8-sig")

    summary_log = build_summary_log(
        listing_before_concat=listing_before_concat,
        listing_after_concat=listing_after_concat,
        sold_before_concat=sold_before_concat,
        sold_after_concat=sold_after_concat,
        listing_before_filter=listing_before_filter,
        listing_after_filter=listing_after_filter,
        sold_before_filter=sold_before_filter,
        sold_after_filter=sold_after_filter,
    )
    summary_log.to_csv(summary_output, index=False, encoding="utf-8-sig")

    write_text_log(
        log_path=txt_log_output,
        start_yyyymm=START_YYYYMM,
        end_yyyymm=end_yyyymm,
        listing_file_counts=listing_file_counts,
        sold_file_counts=sold_file_counts,
        listing_before_concat=listing_before_concat,
        listing_after_concat=listing_after_concat,
        sold_before_concat=sold_before_concat,
        sold_after_concat=sold_after_concat,
        listing_before_filter=listing_before_filter,
        listing_after_filter=listing_after_filter,
        sold_before_filter=sold_before_filter,
        sold_after_filter=sold_after_filter,
    )

    print("\n" + "=" * 80)
    print("WEEK 1 ROW COUNT SUMMARY")
    print("=" * 80)

    print("\nLISTING DATASET")
    print(f"Sum of monthly file rows before concat: {listing_before_concat:,}")
    print(f"Row count after concat:                {listing_after_concat:,}")
    print(f"Row count before Residential filter:   {listing_before_filter:,}")
    print(f"Row count after Residential filter:    {listing_after_filter:,}")

    print("\nSOLD DATASET")
    print(f"Sum of monthly file rows before concat: {sold_before_concat:,}")
    print(f"Row count after concat:                 {sold_after_concat:,}")
    print(f"Row count before Residential filter:    {sold_before_filter:,}")
    print(f"Row count after Residential filter:     {sold_after_filter:,}")

    print("\nSaved files:")
    print(f"- {listing_output}")
    print(f"- {sold_output}")
    print(f"- {file_catalog_output}")
    print(f"- {listing_counts_output}")
    print(f"- {sold_counts_output}")
    print(f"- {summary_output}")
    print(f"- {txt_log_output}")


if __name__ == "__main__":
    main()