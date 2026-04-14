import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(r"C:\Users\User\Desktop\2026 Spring\IDX exchange DA")
INPUT_DIR = BASE_DIR / "Week 2\Week 2-3 Outputs_20260409_144110"   # adjust if needed
OUTPUT_DIR = BASE_DIR / "Week 2\Week 2-3 EDA Charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sold_path = INPUT_DIR / "sold_week2_3_filtered_input.csv"
listing_path = INPUT_DIR / "listing_week2_3_filtered_input.csv"

sold = pd.read_csv(sold_path)
listing = pd.read_csv(listing_path)

# =========================
# Key columns
# =========================
cols = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

# =========================
# Function: Analysis
# =========================
def analyze_dataset(df, name):

    percentile_rows = []
    outlier_rows = []

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        # =========================
        # Histogram
        # =========================
        plt.figure()
        plt.hist(s, bins=50)
        plt.title(f"{name} - {col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(OUTPUT_DIR / f"{name}_{col}_hist.png")
        plt.close()

        # =========================
        # Boxplot
        # =========================
        plt.figure()
        plt.boxplot(s, vert=True)
        plt.title(f"{name} - {col} Boxplot")
        plt.savefig(OUTPUT_DIR / f"{name}_{col}_box.png")
        plt.close()

        # =========================
        # Percentiles
        # =========================
        percentiles = {
            "dataset": name,
            "column": col,
            "min": s.min(),
            "p1": s.quantile(0.01),
            "p5": s.quantile(0.05),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "mean": s.mean(),
            "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
        }

        percentile_rows.append(percentiles)

        # =========================
        # Outlier Detection (IQR)
        # =========================
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count = ((s < lower) | (s > upper)).sum()

        outlier_rows.append({
            "dataset": name,
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outlier_count),
            "total_count": int(len(s)),
            "outlier_pct": outlier_count / len(s)
        })

    return pd.DataFrame(percentile_rows), pd.DataFrame(outlier_rows)


# =========================
# Run for both datasets
# =========================
sold_percentiles, sold_outliers = analyze_dataset(sold, "Sold")
listing_percentiles, listing_outliers = analyze_dataset(listing, "Listing")

# =========================
# Save results
# =========================
percentiles_all = pd.concat([sold_percentiles, listing_percentiles])
outliers_all = pd.concat([sold_outliers, listing_outliers])

percentiles_all.to_csv(OUTPUT_DIR / "percentile_summary.csv", index=False)
outliers_all.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

print("Done. Outputs saved to:", OUTPUT_DIR)