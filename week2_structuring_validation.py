#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================================================
# User settings
# =========================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA")
INPUT_DIR = BASE_DIR / "Week 1\Week 1_version 3"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / f"Week 2-3 Outputs_{RUN_TAG}"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

KEY_NUMERIC_COLS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
]

REQUIRED_SUMMARY_COLS = ["ClosePrice", "LivingArea", "DaysOnMarket"]


# =========================================================
# Helpers
# =========================================================
def find_latest_file(folder: Path, prefix: str) -> Path:
    files = list(folder.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No file found with prefix: {prefix} in {folder}")
    return max(files, key=lambda x: x.stat().st_mtime)


SOLD_INPUT = find_latest_file(INPUT_DIR, "CRMLSSoldCombined_Residential_202401_to_202603")
LISTING_INPUT = find_latest_file(INPUT_DIR, "CRMLSListingCombined_Residential_202401_to_202603")

def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e

    raise ValueError(f"Unable to read CSV: {path}\nLast error: {last_error}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shape_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
            }
        ]
    )


def dtype_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "dataset": dataset_name,
                "column_name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def property_type_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "PropertyType" not in df.columns:
        return pd.DataFrame(
            [{"dataset": dataset_name, "PropertyType": "COLUMN_NOT_FOUND", "count": len(df), "pct": 1.0}]
        )

    counts = df["PropertyType"].fillna("<<MISSING>>").value_counts(dropna=False).reset_index()
    counts.columns = ["PropertyType", "count"]
    counts["dataset"] = dataset_name
    counts["pct"] = counts["count"] / counts["count"].sum()
    return counts[["dataset", "PropertyType", "count", "pct"]]


def null_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "column_name": df.columns,
            "null_count": df.isna().sum().values,
            "null_pct": df.isna().mean().values,
        }
    )
    out["dataset"] = dataset_name
    out["flag_gt_90pct_null"] = out["null_pct"] > 0.90
    out["flag_gt_50pct_null"] = out["null_pct"] > 0.50
    return out.sort_values(["null_pct", "column_name"], ascending=[False, True]).reset_index(drop=True)


def numeric_distribution_summary(df: pd.DataFrame, dataset_name: str, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": col,
                    "available": False,
                    "non_null_count": 0,
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "mean": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "std": np.nan,
                }
            )
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "available": True,
                "non_null_count": int(s.notna().sum()),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "p05": s.quantile(0.05),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "mean": s.mean(),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "std": s.std(),
            }
        )
    return pd.DataFrame(rows)


def county_median_price(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "CountyOrParish" not in df.columns or "ClosePrice" not in df.columns:
        return pd.DataFrame(columns=["dataset", "CountyOrParish", "median_close_price", "transaction_count"])

    tmp = df.copy()
    tmp["ClosePrice"] = pd.to_numeric(tmp["ClosePrice"], errors="coerce")

    out = (
        tmp.groupby("CountyOrParish", dropna=False)["ClosePrice"]
        .agg(median_close_price="median", transaction_count="count")
        .reset_index()
        .sort_values(["median_close_price", "transaction_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    out["dataset"] = dataset_name
    return out[["dataset", "CountyOrParish", "median_close_price", "transaction_count"]]


def sold_above_below_list_summary(sold: pd.DataFrame) -> pd.DataFrame:
    if "ClosePrice" not in sold.columns or "ListPrice" not in sold.columns:
        return pd.DataFrame(
            [{"metric": "status", "value": "ClosePrice or ListPrice column not found"}]
        )

    tmp = sold.copy()
    tmp["ClosePrice"] = pd.to_numeric(tmp["ClosePrice"], errors="coerce")
    tmp["ListPrice"] = pd.to_numeric(tmp["ListPrice"], errors="coerce")

    valid = tmp[(tmp["ClosePrice"].notna()) & (tmp["ListPrice"].notna()) & (tmp["ListPrice"] > 0)].copy()
    valid["ratio_close_to_list"] = valid["ClosePrice"] / valid["ListPrice"]

    above_pct = (valid["ratio_close_to_list"] > 1).mean()
    below_pct = (valid["ratio_close_to_list"] < 1).mean()
    equal_pct = (valid["ratio_close_to_list"] == 1).mean()

    return pd.DataFrame(
        [
            {"metric": "valid_rows_used", "value": int(len(valid))},
            {"metric": "pct_sold_above_list", "value": above_pct},
            {"metric": "pct_sold_below_list", "value": below_pct},
            {"metric": "pct_sold_at_list", "value": equal_pct},
            {"metric": "median_close_to_list_ratio", "value": valid["ratio_close_to_list"].median()},
            {"metric": "mean_close_to_list_ratio", "value": valid["ratio_close_to_list"].mean()},
        ]
    )


def date_consistency_summary(sold: pd.DataFrame) -> pd.DataFrame:
    needed = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
    if not all(c in sold.columns for c in needed):
        return pd.DataFrame(
            [{"metric": "status", "value": "One or more date columns not found"}]
        )

    tmp = sold.copy()
    for c in needed:
        tmp[c] = pd.to_datetime(tmp[c], errors="coerce")

    listing_after_close = (
        (tmp["ListingContractDate"].notna()) &
        (tmp["CloseDate"].notna()) &
        (tmp["ListingContractDate"] > tmp["CloseDate"])
    )

    purchase_after_close = (
        (tmp["PurchaseContractDate"].notna()) &
        (tmp["CloseDate"].notna()) &
        (tmp["PurchaseContractDate"] > tmp["CloseDate"])
    )

    negative_timeline = (
        (tmp["ListingContractDate"].notna()) &
        (tmp["PurchaseContractDate"].notna()) &
        (tmp["ListingContractDate"] > tmp["PurchaseContractDate"])
    )

    return pd.DataFrame(
        [
            {"metric": "listing_after_close_count", "value": int(listing_after_close.sum())},
            {"metric": "purchase_after_close_count", "value": int(purchase_after_close.sum())},
            {"metric": "negative_timeline_count", "value": int(negative_timeline.sum())},
        ]
    )


def add_year_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        out["year_month"] = pd.NaT
        return out

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year_month"] = out[date_col].dt.to_period("M")
    return out


def fetch_mortgage_monthly() -> pd.DataFrame:
    mortgage = pd.read_csv(FRED_URL)

    # handle either possible date column name
    if "DATE" in mortgage.columns:
        mortgage["date"] = pd.to_datetime(mortgage["DATE"], errors="coerce")
    elif "observation_date" in mortgage.columns:
        mortgage["date"] = pd.to_datetime(mortgage["observation_date"], errors="coerce")
    else:
        raise KeyError(f"Could not find date column in FRED data. Columns found: {list(mortgage.columns)}")

    # handle either possible rate column name
    if "MORTGAGE30US" in mortgage.columns:
        mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["MORTGAGE30US"], errors="coerce")
    elif "value" in mortgage.columns:
        mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["value"], errors="coerce")
    else:
        possible_rate_cols = [c for c in mortgage.columns if c not in ["DATE", "observation_date", "date"]]
        raise KeyError(f"Could not find mortgage rate column. Columns found: {list(mortgage.columns)}")

    mortgage = mortgage[["date", "rate_30yr_fixed"]].copy()
    mortgage = mortgage.dropna(subset=["date"])

    mortgage["year_month"] = mortgage["date"].dt.to_period("M")

    mortgage_monthly = (
        mortgage.groupby("year_month", as_index=False)["rate_30yr_fixed"]
        .mean()
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    return mortgage_monthly


def merge_mortgage(df: pd.DataFrame, mortgage_monthly: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col = "CloseDate" if dataset_name.lower() == "sold" else "ListingContractDate"
    out = add_year_month(df, date_col=date_col)
    merged = out.merge(mortgage_monthly, on="year_month", how="left")

    validation = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "join_key_date_column": date_col,
                "null_rate_30yr_fixed_count": int(merged["rate_30yr_fixed"].isna().sum()),
                "total_rows": int(len(merged)),
            }
        ]
    )
    return merged, validation


def build_business_summary_text(
    sold_shape: pd.DataFrame,
    listing_shape: pd.DataFrame,
    sold_prop: pd.DataFrame,
    listing_prop: pd.DataFrame,
    sold_num_req: pd.DataFrame,
    sold_above_below: pd.DataFrame,
    sold_date_consistency: pd.DataFrame,
    county_prices: pd.DataFrame,
) -> str:
    def get_metric(df: pd.DataFrame, name: str):
        row = df[df["metric"] == name]
        return None if row.empty else row["value"].iloc[0]

    sold_rows = sold_shape["row_count"].iloc[0]
    sold_cols = sold_shape["column_count"].iloc[0]
    listing_rows = listing_shape["row_count"].iloc[0]
    listing_cols = listing_shape["column_count"].iloc[0]

    closeprice_row = sold_num_req[sold_num_req["column"] == "ClosePrice"]
    dom_row = sold_num_req[sold_num_req["column"] == "DaysOnMarket"]

    county_top = county_prices.head(5)

    lines = []
    lines.append("IDX Exchange Week 2-3 EDA Summary")
    lines.append("=" * 60)
    lines.append(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("1. Dataset structure")
    lines.append(f"- Sold dataset shape: {sold_rows:,} rows x {sold_cols:,} columns")
    lines.append(f"- Listing dataset shape: {listing_rows:,} rows x {listing_cols:,} columns")
    lines.append("")
    lines.append("2. Property type review")
    lines.append("- These Week 2-3 inputs are already Residential-filtered from Week 1.")
    lines.append(f"- Sold PropertyType values found: {', '.join(sold_prop['PropertyType'].astype(str).head(10).tolist())}")
    lines.append(f"- Listing PropertyType values found: {', '.join(listing_prop['PropertyType'].astype(str).head(10).tolist())}")
    lines.append("")
    lines.append("3. Key numeric findings (Sold)")
    if not closeprice_row.empty:
        lines.append(
            f"- ClosePrice median = {closeprice_row['median'].iloc[0]:,.2f}, "
            f"mean = {closeprice_row['mean'].iloc[0]:,.2f}, "
            f"min = {closeprice_row['min'].iloc[0]:,.2f}, "
            f"max = {closeprice_row['max'].iloc[0]:,.2f}"
        )
    if not dom_row.empty:
        lines.append(
            f"- DaysOnMarket median = {dom_row['median'].iloc[0]:,.2f}, "
            f"mean = {dom_row['mean'].iloc[0]:,.2f}, "
            f"p95 = {dom_row['p95'].iloc[0]:,.2f}, "
            f"max = {dom_row['max'].iloc[0]:,.2f}"
        )
    lines.append("")
    lines.append("4. Sold above vs below list")
    lines.append(f"- % sold above list: {get_metric(sold_above_below, 'pct_sold_above_list'):.2%}")
    lines.append(f"- % sold below list: {get_metric(sold_above_below, 'pct_sold_below_list'):.2%}")
    lines.append(f"- % sold at list: {get_metric(sold_above_below, 'pct_sold_at_list'):.2%}")
    lines.append(f"- Median close-to-list ratio: {get_metric(sold_above_below, 'median_close_to_list_ratio'):.4f}")
    lines.append("")
    lines.append("5. Date consistency checks (Sold)")
    lines.append(f"- listing_after_close_count: {int(get_metric(sold_date_consistency, 'listing_after_close_count')):,}")
    lines.append(f"- purchase_after_close_count: {int(get_metric(sold_date_consistency, 'purchase_after_close_count')):,}")
    lines.append(f"- negative_timeline_count: {int(get_metric(sold_date_consistency, 'negative_timeline_count')):,}")
    lines.append("")
    lines.append("6. Counties with highest median ClosePrice")
    for row in county_top.itertuples(index=False):
        lines.append(
            f"- {row.CountyOrParish}: median={row.median_close_price:,.2f}, count={row.transaction_count:,}"
        )
    lines.append("")
    lines.append("7. Why this task matters")
    lines.append("- Week 2-3 validates whether the Residential dataset is trustworthy before cleaning and feature engineering.")
    lines.append("- Missingness review helps identify weak fields and which columns may need exclusion later.")
    lines.append("- Numeric summaries reveal skew, suspicious values, and likely outliers for Week 4-7 handling.")
    lines.append("- Mortgage enrichment adds an external market context variable for later trend analysis.")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not SOLD_INPUT.exists():
        raise FileNotFoundError(f"Sold input file not found: {SOLD_INPUT}")
    if not LISTING_INPUT.exists():
        raise FileNotFoundError(f"Listing input file not found: {LISTING_INPUT}")

    sold = safe_read_csv(SOLD_INPUT)
    listings = safe_read_csv(LISTING_INPUT)

    # -----------------------------------------------------
    # Core summaries
    # -----------------------------------------------------
    sold_shape = shape_summary(sold, "Sold")
    listing_shape = shape_summary(listings, "Listing")
    shape_df = pd.concat([sold_shape, listing_shape], ignore_index=True)

    sold_dtypes = dtype_summary(sold, "Sold")
    listing_dtypes = dtype_summary(listings, "Listing")
    dtype_df = pd.concat([sold_dtypes, listing_dtypes], ignore_index=True)

    sold_prop = property_type_summary(sold, "Sold")
    listing_prop = property_type_summary(listings, "Listing")
    property_type_df = pd.concat([sold_prop, listing_prop], ignore_index=True)

    sold_nulls = null_summary(sold, "Sold")
    listing_nulls = null_summary(listings, "Listing")
    null_df = pd.concat([sold_nulls, listing_nulls], ignore_index=True)
    high_missing_df = null_df[null_df["flag_gt_90pct_null"]].copy()

    sold_numeric = numeric_distribution_summary(sold, "Sold", KEY_NUMERIC_COLS)
    listing_numeric = numeric_distribution_summary(listings, "Listing", KEY_NUMERIC_COLS)
    numeric_df = pd.concat([sold_numeric, listing_numeric], ignore_index=True)

    sold_numeric_required = numeric_distribution_summary(sold, "Sold", REQUIRED_SUMMARY_COLS)

    county_prices = county_median_price(sold, "Sold")
    sold_above_below = sold_above_below_list_summary(sold)
    sold_date_consistency = date_consistency_summary(sold)

    # -----------------------------------------------------
    # Mortgage enrichment
    # -----------------------------------------------------
    mortgage_monthly = fetch_mortgage_monthly()

    sold_enriched, sold_merge_validation = merge_mortgage(sold, mortgage_monthly, "Sold")
    listing_enriched, listing_merge_validation = merge_mortgage(listings, mortgage_monthly, "Listing")
    merge_validation_df = pd.concat([sold_merge_validation, listing_merge_validation], ignore_index=True)

    # -----------------------------------------------------
    # Save filtered/enriched outputs
    # -----------------------------------------------------
    sold_filtered_output = OUTPUT_DIR / "sold_week2_3_filtered_input.csv"
    listing_filtered_output = OUTPUT_DIR / "listing_week2_3_filtered_input.csv"
    sold_enriched_output = OUTPUT_DIR / "sold_week2_3_with_mortgage.csv"
    listing_enriched_output = OUTPUT_DIR / "listing_week2_3_with_mortgage.csv"

    sold.to_csv(sold_filtered_output, index=False, encoding="utf-8-sig")
    listings.to_csv(listing_filtered_output, index=False, encoding="utf-8-sig")
    sold_enriched.to_csv(sold_enriched_output, index=False, encoding="utf-8-sig")
    listing_enriched.to_csv(listing_enriched_output, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------
    shape_df.to_csv(OUTPUT_DIR / "dataset_shape_summary.csv", index=False, encoding="utf-8-sig")
    dtype_df.to_csv(OUTPUT_DIR / "column_dtype_summary.csv", index=False, encoding="utf-8-sig")
    property_type_df.to_csv(OUTPUT_DIR / "property_type_summary.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(OUTPUT_DIR / "null_summary.csv", index=False, encoding="utf-8-sig")
    high_missing_df.to_csv(OUTPUT_DIR / "high_missing_columns_gt_90pct.csv", index=False, encoding="utf-8-sig")
    numeric_df.to_csv(OUTPUT_DIR / "numeric_distribution_summary.csv", index=False, encoding="utf-8-sig")
    county_prices.to_csv(OUTPUT_DIR / "county_median_close_price.csv", index=False, encoding="utf-8-sig")
    sold_above_below.to_csv(OUTPUT_DIR / "sold_above_below_list_summary.csv", index=False, encoding="utf-8-sig")
    sold_date_consistency.to_csv(OUTPUT_DIR / "sold_date_consistency_summary.csv", index=False, encoding="utf-8-sig")
    mortgage_monthly.to_csv(OUTPUT_DIR / "mortgage_monthly_from_fred.csv", index=False, encoding="utf-8-sig")
    merge_validation_df.to_csv(OUTPUT_DIR / "mortgage_merge_validation.csv", index=False, encoding="utf-8-sig")

    business_summary = build_business_summary_text(
        sold_shape=sold_shape,
        listing_shape=listing_shape,
        sold_prop=sold_prop,
        listing_prop=listing_prop,
        sold_num_req=sold_numeric_required,
        sold_above_below=sold_above_below,
        sold_date_consistency=sold_date_consistency,
        county_prices=county_prices,
    )
    (OUTPUT_DIR / "week2_3_eda_summary.txt").write_text(business_summary, encoding="utf-8-sig")

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------
    print("=" * 80)
    print("WEEK 2-3 STRUCTURING / VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Output folder: {OUTPUT_DIR}")
    print("")
    print("Saved core outputs:")
    print(f"- {sold_filtered_output.name}")
    print(f"- {listing_filtered_output.name}")
    print(f"- {sold_enriched_output.name}")
    print(f"- {listing_enriched_output.name}")
    print("")
    print("Saved report files:")
    print("- dataset_shape_summary.csv")
    print("- column_dtype_summary.csv")
    print("- property_type_summary.csv")
    print("- null_summary.csv")
    print("- high_missing_columns_gt_90pct.csv")
    print("- numeric_distribution_summary.csv")
    print("- county_median_close_price.csv")
    print("- sold_above_below_list_summary.csv")
    print("- sold_date_consistency_summary.csv")
    print("- mortgage_monthly_from_fred.csv")
    print("- mortgage_merge_validation.csv")
    print("- week2_3_eda_summary.txt")


if __name__ == "__main__":
    main()