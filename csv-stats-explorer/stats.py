"""
stats.py — Core statistical analysis module for CSV Stats Explorer

Handles: data loading, cleaning, descriptive stats, outlier detection,
distribution analysis, and correlation. All functions are pure — they
accept DataFrames and return results, keeping them testable independently
of the Streamlit UI.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


# ── Loading & Cleaning ────────────────────────────────────────────────────────

def load_csv(filepath_or_buffer) -> pd.DataFrame:
    """Load CSV, infer dtypes, return raw DataFrame."""
    return pd.read_csv(filepath_or_buffer)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_column_summary(df: pd.DataFrame, col: str) -> dict:
    """
    Per-column metadata: dtype, missing count/pct, unique count, sample values.
    """
    series = df[col]
    return {
        "dtype":       str(series.dtype),
        "n_total":     len(series),
        "n_missing":   int(series.isna().sum()),
        "pct_missing": round(series.isna().mean() * 100, 2),
        "n_unique":    int(series.nunique()),
        "sample":      series.dropna().head(5).tolist(),
    }


# ── Descriptive Statistics ────────────────────────────────────────────────────

def descriptive_stats(series: pd.Series) -> pd.DataFrame:
    """
    Full five-number summary + mean, stddev, variance, skewness, kurtosis.
    Returns a single-row DataFrame for display.
    """
    s = series.dropna()
    if len(s) == 0:
        return pd.DataFrame()

    result = {
        "Count":     len(s),
        "Mean":      round(s.mean(), 4),
        "Std Dev":   round(s.std(ddof=1), 4),
        "Variance":  round(s.var(ddof=1), 4),
        "Skewness":  round(float(scipy_stats.skew(s)), 4),
        "Kurtosis":  round(float(scipy_stats.kurtosis(s)), 4),
        "Min":       round(s.min(), 4),
        "Q1 (25%)":  round(s.quantile(0.25), 4),
        "Median":    round(s.median(), 4),
        "Q3 (75%)":  round(s.quantile(0.75), 4),
        "Max":       round(s.max(), 4),
        "IQR":       round(s.quantile(0.75) - s.quantile(0.25), 4),
        "Range":     round(s.max() - s.min(), 4),
    }
    return pd.DataFrame([result]).T.rename(columns={0: "Value"})


def percentile_table(series: pd.Series,
                     percentiles: list[float] = None) -> pd.DataFrame:
    """Compute arbitrary percentile table."""
    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    s = series.dropna()
    rows = [{"Percentile": f"P{p}", "Value": round(s.quantile(p / 100), 4)}
            for p in percentiles]
    return pd.DataFrame(rows)


# ── Outlier Detection ─────────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series,
                         k: float = 1.5) -> dict:
    """
    IQR method: outliers lie below Q1 - k*IQR or above Q3 + k*IQR.
    Returns indices, values, and summary statistics.
    """
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (s < lower) | (s > upper)
    outlier_series = s[mask]

    return {
        "method":      f"IQR (k={k})",
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
        "n_outliers":  int(mask.sum()),
        "pct_outliers": round(mask.mean() * 100, 2),
        "outlier_values": outlier_series.values.tolist(),
        "outlier_indices": outlier_series.index.tolist(),
    }


def detect_outliers_zscore(series: pd.Series,
                            threshold: float = 3.0) -> dict:
    """
    Z-score method: |z| > threshold flags an outlier.
    """
    s = series.dropna()
    z = np.abs((s - s.mean()) / s.std(ddof=1))
    mask = z > threshold

    return {
        "method":       f"Z-score (|z| > {threshold})",
        "threshold":    threshold,
        "n_outliers":   int(mask.sum()),
        "pct_outliers": round(mask.mean() * 100, 2),
        "outlier_values": s[mask].values.tolist(),
    }


# ── Normality Testing ─────────────────────────────────────────────────────────

def normality_test(series: pd.Series) -> dict:
    """
    Shapiro-Wilk (n ≤ 5000) or D'Agostino-Pearson test.
    Returns test name, statistic, p-value, and interpretation.
    """
    s = series.dropna()
    n = len(s)
    if n < 3:
        return {"test": "N/A", "p_value": None, "normal": None}

    if n <= 5000:
        stat, p = scipy_stats.shapiro(s)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = scipy_stats.normaltest(s)
        test_name = "D'Agostino-Pearson"

    return {
        "test":      test_name,
        "statistic": round(float(stat), 6),
        "p_value":   round(float(p), 6),
        "normal":    bool(p > 0.05),
        "interpretation": (
            f"Data appears normal (p={p:.4f} > 0.05)"
            if p > 0.05
            else f"Data is not normal (p={p:.4f} ≤ 0.05)"
        ),
    }


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_with_target(df: pd.DataFrame,
                              target_col: str) -> pd.DataFrame:
    """
    Pearson and Spearman correlations of all numeric columns with target_col.
    Returns a sorted DataFrame.
    """
    numeric_cols = [c for c in get_numeric_columns(df) if c != target_col]
    target = df[target_col].dropna()

    rows = []
    for col in numeric_cols:
        aligned = df[[target_col, col]].dropna()
        if len(aligned) < 3:
            continue
        r_p, p_p = scipy_stats.pearsonr(aligned[target_col], aligned[col])
        r_s, p_s = scipy_stats.spearmanr(aligned[target_col], aligned[col])
        rows.append({
            "Column":          col,
            "Pearson r":       round(r_p, 4),
            "Pearson p":       round(p_p, 4),
            "Spearman r":      round(r_s, 4),
            "Spearman p":      round(p_s, 4),
            "|Pearson r|":     round(abs(r_p), 4),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("|Pearson r|", ascending=False)
    return result.drop(columns=["|Pearson r|"])


def full_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for all numeric columns."""
    return df[get_numeric_columns(df)].corr()


# ── Missing Value Report ──────────────────────────────────────────────────────

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of missing values per column, sorted by count descending."""
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        rows.append({
            "Column":      col,
            "Type":        str(df[col].dtype),
            "Missing":     n_miss,
            "Missing (%)": round(n_miss / len(df) * 100, 2),
            "Present":     len(df) - n_miss,
        })
    return (pd.DataFrame(rows)
              .sort_values("Missing", ascending=False)
              .reset_index(drop=True))
