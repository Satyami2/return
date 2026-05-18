"""
Mutual Fund Analysis Dashboard
==============================
Streamlit app for analyzing Indian mutual funds across categories,
comparing fund performance against indices, and inspecting rolling returns.

Run:
    streamlit run mf_analysis_app.py

Default: all .xlsx files sit next to this script.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

st.set_page_config(page_title="MF Analysis", page_icon="📈", layout="wide")

# Light global styling — tightens vertical rhythm and softens default streamlit blocks
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        section[data-testid="stSidebar"] { background-color: #fafafa; }
        h1 { font-size: 1.8rem !important; margin-bottom: 0.2rem !important; }
        h2 { font-size: 1.15rem !important; margin-top: 1.2rem !important; }
        h3 { font-size: 1.0rem !important; margin-top: 1rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA_DIR = "."

CATEGORY_FILES = {
    "Large Cap":          ["largecap1.xlsx", "largecap2.xlsx"],
    "Large & Mid Cap":    ["largeandmidcapa.xlsx"],
    "Mid Cap":            ["midcap.xlsx"],
    "Small Cap":          ["smallcap.xlsx"],
    "Flexi Cap":          ["flexicap1.xlsx", "flexicap2.xlsx"],
    "Multi Cap":          ["multicap.xlsx"],
}

INDEX_FILES = {
    "NIFTY 50":            "nifty50.xlsx",
    "NIFTY 500":           "nifty500.xlsx",
    "Nifty Midcap 100":    "nifty_midcap100.xlsx",
    "Nifty Smallcap 100":  "niftysmallcap100.xlsx",
}

CATEGORY_DEFAULT_INDEX = {
    "Large Cap":         "NIFTY 50",
    "Large & Mid Cap":   "NIFTY 500",
    "Mid Cap":           "Nifty Midcap 100",
    "Small Cap":         "Nifty Smallcap 100",
    "Flexi Cap":         "NIFTY 500",
    "Multi Cap":         "NIFTY 500",
}

# Period -> days. Order matters for display.
PERIOD_DAYS = {
    "1M":  30,
    "3M":  91,
    "6M":  182,
    "1Y":  365,
    "3Y":  365 * 3,
    "5Y":  365 * 5,
    "10Y": 365 * 10,
}

# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fund_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=2, skiprows=[3])
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_index_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=2)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Close Price"] = pd.to_numeric(df["Close Price"], errors="coerce")
    df = df.dropna(subset=["Close Price"]).reset_index(drop=True)
    return df[["Date", "Close Price"]].sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=True)
def load_category(data_dir: str, category: str) -> pd.DataFrame:
    """Load and merge all fund files for a category. Tolerates missing files
    (e.g. only largecap1.xlsx present without largecap2.xlsx)."""
    files = CATEGORY_FILES[category]
    frames = []
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            frames.append(load_fund_file(path))
    if not frames:
        return pd.DataFrame(columns=["Date"])
    df = frames[0]
    for nxt in frames[1:]:
        df = pd.merge(df, nxt, on="Date", how="outer")
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_index(data_dir: str, index_name: str) -> pd.DataFrame:
    return load_index_file(os.path.join(data_dir, INDEX_FILES[index_name]))


@st.cache_data(show_spinner=False)
def get_fund_universe(data_dir: str) -> dict:
    """{fund_name: category} across all categories."""
    universe = {}
    for cat in CATEGORY_FILES:
        try:
            df = load_category(data_dir, cat)
            for fund in df.columns[1:]:
                universe[fund] = cat
        except Exception:
            pass
    return universe


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def get_series(data_dir: str, name: str, kind: str) -> pd.DataFrame:
    """Return [Date, Value] for a fund or index."""
    if kind == "index":
        return load_index(data_dir, name).rename(columns={"Close Price": "Value"})
    cat = get_fund_universe(data_dir).get(name)
    if cat is None:
        return pd.DataFrame(columns=["Date", "Value"])
    df = load_category(data_dir, cat)
    return df[["Date", name]].dropna().rename(columns={name: "Value"}).reset_index(drop=True)


def cagr(start_val, end_val, years):
    if start_val is None or end_val is None or start_val <= 0 or end_val <= 0 or years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def value_near(series: pd.DataFrame, target_date: pd.Timestamp, tolerance_days: int = 7):
    """Last value at or before target_date, falling back to closest within tolerance."""
    if series.empty:
        return None, None
    s = series.sort_values("Date").reset_index(drop=True)
    at_or_before = s[s["Date"] <= target_date]
    if not at_or_before.empty:
        row = at_or_before.iloc[-1]
        if (target_date - row["Date"]).days <= tolerance_days * 5:
            return row["Date"], row["Value"]
    s = s.assign(diff=(s["Date"] - target_date).abs())
    row = s.loc[s["diff"].idxmin()]
    if row["diff"].days <= tolerance_days:
        return row["Date"], row["Value"]
    return None, None


def period_return(series: pd.DataFrame, period_label: str, as_of: pd.Timestamp):
    """Absolute return for periods <=1Y, CAGR for >1Y."""
    days = PERIOD_DAYS[period_label]
    target_start = as_of - pd.Timedelta(days=days)
    _, end_val = value_near(series, as_of)
    _, start_val = value_near(series, target_start)
    if start_val is None or end_val is None:
        return np.nan
    years = days / 365.0
    if years > 1:
        return cagr(start_val, end_val, years)
    return (end_val / start_val) - 1


def rolling_returns(series: pd.DataFrame, window_years: int) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["Date", "Return"])
    s = series.sort_values("Date").set_index("Date").asfreq("D").ffill()
    window_days = int(round(window_years * 365.25))
    shifted = s["Value"].shift(window_days)
    cagr_series = (s["Value"] / shifted) ** (1 / window_years) - 1
    return pd.DataFrame({"Date": s.index, "Return": cagr_series.values}).dropna().reset_index(drop=True)


def normalize_to_100(series: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    s = series[series["Date"] >= start_date].sort_values("Date").reset_index(drop=True)
    if s.empty:
        return s
    base = s["Value"].iloc[0]
    if base == 0 or pd.isna(base):
        return pd.DataFrame(columns=["Date", "Value"])
    s = s.copy()
    s["Value"] = s["Value"] / base * 100.0
    return s


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

def fund_picker(label_prefix: str, universe: dict, key_prefix: str,
                max_funds=None, default_count: int = 2):
    """Render Category filter + Fund multiselect. Returns list of fund names."""
    cat_options = ["All categories"] + list(CATEGORY_FILES.keys())
    cat = st.selectbox(f"{label_prefix} — category filter",
                       options=cat_options, index=0, key=f"{key_prefix}_cat")

    if cat == "All categories":
        filtered = sorted(universe.keys())
    else:
        filtered = sorted([f for f, c in universe.items() if c == cat])

    if not filtered:
        st.info("No funds in this category.")
        return []

    default = filtered[:default_count] if len(filtered) >= default_count else filtered
    kwargs = dict(
        options=filtered,
        default=default,
        key=f"{key_prefix}_sel",
        help="Type to search. Use the category filter above to narrow.",
    )
    if max_funds is not None:
        kwargs["max_selections"] = max_funds
    return st.multiselect(label_prefix, **kwargs)


def latest_inception(data_dir: str, fund_names: list):
    """Most recent inception date across selected funds. Anchors the chart."""
    starts = []
    for f in fund_names:
        s = get_series(data_dir, f, "fund")
        if not s.empty:
            starts.append(s["Date"].min())
    return max(starts) if starts else None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("📈 MF Analysis")
data_dir = st.sidebar.text_input("Data folder", value=DEFAULT_DATA_DIR,
                                 help="Folder containing the .xlsx files")

if not os.path.isdir(data_dir):
    st.error(f"Data folder `{data_dir}` not found. Update the path in the sidebar.")
    st.stop()

page = st.sidebar.radio(
    "Analysis",
    ["Performance Graph", "Rolling Returns", "Category Leaderboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Returns from Adjusted NAVs. Periods > 1Y are CAGR; ≤ 1Y are absolute. "
    "NAV gaps are forward-filled for rolling computations."
)

with st.spinner("Indexing fund universe..."):
    universe = get_fund_universe(data_dir)

index_options = list(INDEX_FILES.keys())


# ============================================================================
# PAGE 1 — Performance Graph
# ============================================================================

if page == "Performance Graph":
    st.title("Performance Comparison")
    st.caption("Compare funds and indices rebased to 100. The chart auto-starts "
               "from the most recent fund's inception so every line begins together.")

    sel_col, date_col = st.columns([3, 2], gap="large")

    with sel_col:
        selected_funds = fund_picker("Funds", universe, key_prefix="p1",
                                     default_count=2)
        selected_indices = st.multiselect(
            "Indices", options=index_options, default=["NIFTY 50"], key="p1_idx",
        )

    today = date.today()
    auto_start = latest_inception(data_dir, selected_funds)
    suggested_start = auto_start.date() if auto_start is not None \
        else today - timedelta(days=365 * 3)

    with date_col:
        use_auto = st.toggle(
            "Auto-start from most recent fund",
            value=True,
            help="When on, the chart begins on the inception date of the latest fund selected.",
        )
        if use_auto and auto_start is not None:
            start_date = suggested_start
            st.text_input("Start date", value=str(suggested_start),
                          disabled=True, key="p1_sd_disp")
        else:
            start_date = st.date_input(
                "Start date", value=suggested_start,
                min_value=date(1990, 1, 1), max_value=today, key="p1_sd",
            )
        end_date = st.date_input(
            "End date", value=today,
            min_value=start_date, max_value=today, key="p1_ed",
        )

    if not selected_funds and not selected_indices:
        st.info("Pick at least one fund or index above.")
        st.stop()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    items = [(f, "fund") for f in selected_funds] + \
            [(i, "index") for i in selected_indices]

    plot_rows, period_rows = [], []
    for name, kind in items:
        s_full = get_series(data_dir, name, kind)
        s = s_full[(s_full["Date"] >= start_ts) & (s_full["Date"] <= end_ts)]
        if s.empty:
            continue
        rebased = normalize_to_100(s, start_ts)
        rebased["Name"] = name
        rebased["Type"] = kind.capitalize()
        plot_rows.append(rebased)

        # Period returns table: compute against full series and end_ts so 1M/3M/etc.
        # are as-of today, regardless of where the chart starts.
        prow = {"Name": name, "Type": kind.capitalize()}
        for plabel in PERIOD_DAYS:
            r = period_return(s_full, plabel, end_ts)
            prow[plabel] = round(r * 100, 2) if pd.notna(r) else None
        period_rows.append(prow)

    if not plot_rows:
        st.warning("No data in the selected range.")
        st.stop()

    plot_df = pd.concat(plot_rows, ignore_index=True)
    fig = px.line(plot_df, x="Date", y="Value", color="Name",
                  line_dash="Type", labels={"Value": "Rebased to 100"})
    fig.update_layout(
        height=500, hovermode="x unified", legend_title="",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    fig.add_hline(y=100, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Period returns (as of {end_ts.date()})")
    st.caption("≤ 1 year: absolute return. > 1 year: CAGR. Values in %.")

    pdf = pd.DataFrame(period_rows)
    cols = ["Name", "Type"] + list(PERIOD_DAYS.keys())
    pdf = pdf[cols]
    numeric_cols = list(PERIOD_DAYS.keys())
    styled = (
        pdf.style
        .format({c: "{:.2f}" for c in numeric_cols}, na_rep="—")
        .background_gradient(subset=numeric_cols, cmap="RdYlGn", axis=0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 2 — Rolling Returns
# ============================================================================

elif page == "Rolling Returns":
    st.title("Rolling Returns")
    st.caption("Daily rolling CAGR over a chosen window. Compare 1Y, 3Y, or 5Y "
               "rolling returns across funds and an index.")

    sel_col, opt_col = st.columns([3, 2], gap="large")

    with sel_col:
        selected_funds = fund_picker("Funds (up to 3)", universe,
                                     key_prefix="p2", max_funds=3,
                                     default_count=2)
        selected_indices = st.multiselect(
            "Indices", options=index_options, default=["NIFTY 50"], key="p2_idx",
        )

    with opt_col:
        window = st.radio("Rolling window", options=[1, 3, 5], index=1,
                          format_func=lambda x: f"{x}-Year", horizontal=True)
        today = date.today()
        default_start = today - timedelta(days=365 * 10)
        start_date = st.date_input("Plot from", value=default_start,
                                   min_value=date(1990, 1, 1),
                                   max_value=today, key="p2_sd")

    if not selected_funds and not selected_indices:
        st.info("Pick at least one fund or index above.")
        st.stop()

    start_ts = pd.Timestamp(start_date)

    items = [(f, "fund") for f in selected_funds] + \
            [(i, "index") for i in selected_indices]

    plot_rows, stat_rows = [], []
    for name, kind in items:
        s = get_series(data_dir, name, kind)
        if s.empty:
            continue
        rr = rolling_returns(s, window)
        rr = rr[rr["Date"] >= start_ts]
        if rr.empty:
            stat_rows.append({"Name": name, "Type": kind.capitalize(),
                              "Note": "Not enough history"})
            continue
        rr["Name"] = name
        rr["Type"] = kind.capitalize()
        rr["Return %"] = rr["Return"] * 100
        plot_rows.append(rr)
        stat_rows.append({
            "Name": name,
            "Type": kind.capitalize(),
            "Mean %":   round(rr["Return %"].mean(), 2),
            "Median %": round(rr["Return %"].median(), 2),
            "Min %":    round(rr["Return %"].min(), 2),
            "Max %":    round(rr["Return %"].max(), 2),
            "Std %":    round(rr["Return %"].std(), 2),
            "% > 0":    round((rr["Return %"] > 0).mean() * 100, 1),
            "Obs":      len(rr),
        })

    if not plot_rows:
        st.warning("No rolling-return data computed for the current selections.")
        st.stop()

    plot_df = pd.concat(plot_rows, ignore_index=True)
    fig = px.line(plot_df, x="Date", y="Return %", color="Name",
                  line_dash="Type",
                  labels={"Return %": f"{window}Y Rolling CAGR (%)"})
    fig.update_layout(
        height=500, hovermode="x unified", legend_title="",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    fig.add_hline(y=0, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    st.subheader("Distribution")
    fig2 = px.histogram(plot_df, x="Return %", color="Name",
                        barmode="overlay", opacity=0.55, nbins=50)
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE 3 — Category Leaderboard
# ============================================================================

elif page == "Category Leaderboard":
    st.title("Category Leaderboard")
    st.caption("Rank every fund in a category by return for a chosen period.")

    c1, c2, c3 = st.columns(3)
    with c1:
        category = st.selectbox("Category", options=list(CATEGORY_FILES.keys()))
    with c2:
        period = st.selectbox("Period", options=list(PERIOD_DAYS.keys()), index=3)
    with c3:
        today = date.today()
        as_of = st.date_input("As of", value=today,
                              min_value=date(1995, 1, 1), max_value=today)

    show_benchmark = st.checkbox(
        f"Show benchmark ({CATEGORY_DEFAULT_INDEX[category]})", value=True
    )

    as_of_ts = pd.Timestamp(as_of)
    cat_df = load_category(data_dir, category)
    funds = [c for c in cat_df.columns if c != "Date"]

    rows = []
    for fund in funds:
        s = cat_df[["Date", fund]].dropna().rename(columns={fund: "Value"})
        if s.empty:
            continue
        ret = period_return(s, period, as_of_ts)
        if not np.isnan(ret):
            rows.append({
                "Fund": fund,
                "Return %": round(ret * 100, 2),
                "Data start": s["Date"].min().strftime("%Y-%m-%d"),
            })

    benchmark_row = None
    if show_benchmark:
        idx_name = CATEGORY_DEFAULT_INDEX[category]
        idx = load_index(data_dir, idx_name).rename(columns={"Close Price": "Value"})
        ret = period_return(idx, period, as_of_ts)
        if not np.isnan(ret):
            benchmark_row = {
                "Fund": f"⭐ {idx_name} (benchmark)",
                "Return %": round(ret * 100, 2),
                "Data start": idx["Date"].min().strftime("%Y-%m-%d"),
            }

    if not rows:
        st.warning("No funds in this category have enough data for the chosen period.")
        st.stop()

    df = pd.DataFrame(rows).sort_values("Return %", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    if benchmark_row is not None:
        bench_pct = benchmark_row["Return %"]
        n_beat = (df["Return %"] > bench_pct).sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Funds with data", len(df))
        m2.metric("Top return %", f"{df['Return %'].iloc[0]:.2f}")
        m3.metric("Benchmark %", f"{bench_pct:.2f}")
        m4.metric("Funds beating benchmark", f"{n_beat} / {len(df)}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Funds with data", len(df))
        m2.metric("Top return %", f"{df['Return %'].iloc[0]:.2f}")
        m3.metric("Median return %", f"{df['Return %'].median():.2f}")

    label = "CAGR %" if PERIOD_DAYS[period] > 365 else "Total Return %"
    plot_df = df.rename(columns={"Return %": label}).copy()
    if benchmark_row is not None:
        plot_df = pd.concat(
            [plot_df,
             pd.DataFrame([{**benchmark_row, "Rank": "—"}]).rename(
                 columns={"Return %": label})],
            ignore_index=True,
        )

    fig = px.bar(
        plot_df.sort_values(label, ascending=True),
        x=label, y="Fund", orientation="h",
        color=label, color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        height=max(420, 24 * len(plot_df)),
        yaxis_title="", coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    if benchmark_row is not None:
        fig.add_vline(x=benchmark_row["Return %"], line_dash="dash",
                      line_color="black", opacity=0.6,
                      annotation_text="Benchmark", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Leaderboard")
    show_df = df.rename(columns={"Return %": label})
    if benchmark_row is not None:
        st.caption(f"Benchmark — **{benchmark_row['Fund']}**: {benchmark_row['Return %']}%")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download leaderboard CSV", data=csv,
                       file_name=f"{category}_{period}_leaderboard.csv",
                       mime="text/csv")
