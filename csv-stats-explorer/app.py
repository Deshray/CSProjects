"""
app.py — CSV Stats Explorer: interactive Streamlit dashboard

Upload any CSV file and explore a selected numeric column:
  • Full descriptive statistics (mean, std, skewness, kurtosis, percentiles)
  • Missing value and data quality report
  • Outlier detection via IQR and Z-score
  • Distribution plots (histogram + KDE, box plot, Q-Q plot)
  • Normality test (Shapiro-Wilk / D'Agostino-Pearson)
  • Correlation with all other numeric columns
  • Full correlation heatmap

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

import stats  # local module

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CSV Stats Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CSV Stats Explorer")
st.caption("Upload a CSV file to explore a numeric column with full descriptive statistics, "
           "outlier detection, distribution analysis, and correlation.")

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = stats.load_csv(uploaded)
st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns**")

# ── Sidebar: column & options ─────────────────────────────────────────────────

numeric_cols = stats.get_numeric_columns(df)
if not numeric_cols:
    st.error("No numeric columns found in this CSV.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    col = st.selectbox("Numeric column to analyse", numeric_cols)
    iqr_k = st.slider("IQR fence multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    z_thresh = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
    n_bins = st.slider("Histogram bins", 5, 100, 30)

series = df[col].dropna()

# ── Data Quality ──────────────────────────────────────────────────────────────

st.header("🔍 Data Quality")
col_info = stats.get_column_summary(df, col)

q1, q2, q3, q4 = st.columns(4)
q1.metric("Total rows",      f"{col_info['n_total']:,}")
q2.metric("Missing values",  f"{col_info['n_missing']:,}  ({col_info['pct_missing']}%)")
q3.metric("Unique values",   f"{col_info['n_unique']:,}")
q4.metric("Data type",       col_info["dtype"])

with st.expander("Missing value report (all columns)"):
    mv = stats.missing_value_report(df)
    st.dataframe(mv[mv["Missing"] > 0] if mv["Missing"].sum() > 0 else mv,
                 use_container_width=True)

# ── Descriptive Statistics ────────────────────────────────────────────────────

st.header("📐 Descriptive Statistics")

desc_df = stats.descriptive_stats(series)
pct_df  = stats.percentile_table(series)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Summary — `{col}`")
    st.dataframe(desc_df, use_container_width=True)
with right:
    st.subheader("Percentile Table")
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

# Normality test
norm = stats.normality_test(series)
if norm["p_value"] is not None:
    icon = "✅" if norm["normal"] else "❌"
    st.info(f"{icon} **{norm['test']}**: {norm['interpretation']}  "
            f"(statistic = {norm['statistic']}, p = {norm['p_value']})")

# ── Distribution Plots ────────────────────────────────────────────────────────

st.header("📈 Distribution")

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Histogram + KDE", "Box Plot", "Q-Q Plot"],
)

# Histogram + KDE
kde_x = np.linspace(series.min(), series.max(), 300)
kde_y = scipy_stats.gaussian_kde(series)(kde_x)

fig.add_trace(go.Histogram(x=series, nbinsx=n_bins, name="Count",
                            marker_color="#4C72B0", opacity=0.7,
                            histnorm="probability density"), row=1, col=1)
fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines",
                          line=dict(color="#DD4444", width=2),
                          name="KDE"), row=1, col=1)

# Box plot
fig.add_trace(go.Box(y=series, name=col, marker_color="#4C72B0",
                     boxpoints="outliers"), row=1, col=2)

# Q-Q plot
(osm, osr), _ = scipy_stats.probplot(series, dist="norm")
fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                          marker=dict(color="#4C72B0", size=4),
                          name="Q-Q"), row=1, col=3)
ql = min(osm[0], osr[0])
qh = max(osm[-1], osr[-1])
fig.add_trace(go.Scatter(x=[ql, qh], y=[ql, qh], mode="lines",
                          line=dict(color="#DD4444", dash="dash"),
                          name="Normal"), row=1, col=3)

fig.update_layout(height=420, showlegend=False,
                  margin=dict(t=50, b=20, l=20, r=20))
st.plotly_chart(fig, use_container_width=True)

# ── Outlier Detection ─────────────────────────────────────────────────────────

st.header("🚨 Outlier Detection")

iqr_res = stats.detect_outliers_iqr(series, k=iqr_k)
z_res   = stats.detect_outliers_zscore(series, threshold=z_thresh)

oc1, oc2 = st.columns(2)

with oc1:
    st.subheader(f"IQR Method (k = {iqr_k})")
    st.metric("Outliers found", f"{iqr_res['n_outliers']}  ({iqr_res['pct_outliers']}%)")
    st.write(f"**Fences:** [{iqr_res['lower_fence']}, {iqr_res['upper_fence']}]")
    if iqr_res["outlier_values"]:
        with st.expander("Outlier values"):
            st.write(sorted(iqr_res["outlier_values"]))

with oc2:
    st.subheader(f"Z-Score Method (|z| > {z_thresh})")
    st.metric("Outliers found", f"{z_res['n_outliers']}  ({z_res['pct_outliers']}%)")
    if z_res["outlier_values"]:
        with st.expander("Outlier values"):
            st.write(sorted(z_res["outlier_values"]))

# Scatter showing outliers
fig_out = go.Figure()
iqr_idx = set(iqr_res["outlier_indices"])
colors = ["#DD4444" if i in iqr_idx else "#4C72B0" for i in series.index]
fig_out.add_trace(go.Scatter(
    x=list(range(len(series))), y=series.values,
    mode="markers",
    marker=dict(color=colors, size=5, opacity=0.7),
    name=col
))
fig_out.add_hline(y=iqr_res["lower_fence"], line_dash="dash",
                   line_color="orange", annotation_text="Lower fence")
fig_out.add_hline(y=iqr_res["upper_fence"], line_dash="dash",
                   line_color="orange", annotation_text="Upper fence")
fig_out.update_layout(height=300, title="Data points (red = IQR outliers)",
                       margin=dict(t=40, b=20, l=20, r=20),
                       xaxis_title="Row index", yaxis_title=col)
st.plotly_chart(fig_out, use_container_width=True)

# ── Correlation ───────────────────────────────────────────────────────────────

if len(numeric_cols) > 1:
    st.header("🔗 Correlation")

    corr_df = stats.correlation_with_target(df, col)
    if not corr_df.empty:
        st.subheader(f"Correlation with `{col}`")
        st.dataframe(corr_df.style.background_gradient(
            subset=["Pearson r"], cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True, hide_index=True)

    st.subheader("Full Correlation Heatmap")
    corr_mat = stats.full_correlation_matrix(df)
    fig_heat = px.imshow(
        corr_mat,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(height=max(300, len(numeric_cols) * 55),
                            margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Raw Data Preview ──────────────────────────────────────────────────────────

with st.expander("📄 Raw data preview"):
    st.dataframe(df.head(100), use_container_width=True)
