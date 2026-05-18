"""
Mutual Fund Analysis Dashboard
==============================
A Streamlit app for analyzing Indian mutual funds across categories,
comparing fund performance against indices, and looking at rolling returns.

Run with:
    streamlit run mf_analysis_app.py

Default: all .xlsx files sit next to this script (same folder). Override the
folder path from the sidebar at runtime if needed.

Expected files:
    flexicap1.xlsx, flexicap2.xlsx
    largecap1.xlsx, largecap2.xlsx
    largeandmidcapa.xlsx
    midcap.xlsx
    multicap.xlsx
    smallcap.xlsx
    nifty50.xlsx
    nifty500.xlsx
    nifty_midcap100.xlsx
    niftysmallcap100.xlsx
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="MF Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_DATA_DIR = "."

# Category -> list of fund excel files (some categories have multiple files)
CATEGORY_FILES = {
    "Large Cap":          ["largecap1.xlsx", "largecap2.xlsx"],
    "Large & Mid Cap":    ["largeandmidcapa.xlsx"],
    "Mid Cap":            ["midcap.xlsx"],
    "Small Cap":          ["smallcap.xlsx"],
    "Flexi Cap":          ["flexicap1.xlsx", "flexicap2.xlsx"],
    "Multi Cap":          ["multicap.xlsx"],
}

# Available indices
INDEX_FILES = {
    "NIFTY 50":            "nifty50.xlsx",
    "NIFTY 500":           "nifty500.xlsx",
    "Nifty Midcap 100":    "nifty_midcap100.xlsx",
    "Nifty Smallcap 100":  "niftysmallcap100.xlsx",
}

# Suggested benchmark index for each category (sensible default)
CATEGORY_DEFAULT_INDEX = {
    "Large Cap":         "NIFTY 50",
    "Large & Mid Cap":   "NIFTY 500",
    "Mid Cap":           "Nifty Midcap 100",
    "Small Cap":         "Nifty Smallcap 100",
    "Flexi Cap":         "NIFTY 500",
    "Multi Cap":         "NIFTY 500",
}

# Standard period definitions (in calendar days)
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
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fund_file(path: str) -> pd.DataFrame:
    """Load a fund NAV excel file.

    Layout: row 1 = ' >>NAV Data' banner, row 2 = blank, row 3 = fund names,
    row 4 = label ('Adjusted NAV NonCorporate(Rs)'), row 5+ = data
    (Date in col A, NAVs in subsequent cols). Last row(s) may contain a
    disclaimer footer, which we drop by filtering non-datetime dates.
    """
    df = pd.read_excel(path, header=2, skiprows=[3])
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_index_file(path: str) -> pd.DataFrame:
    """Load an index file. Layout: Index Name | Date | Close Price."""
    df = pd.read_excel(path, header=2)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Close Price"] = pd.to_numeric(df["Close Price"], errors="coerce")
    df = df.dropna(subset=["Close Price"]).reset_index(drop=True)
    return df[["Date", "Close Price"]].sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=True)
def load_category(data_dir: str, category: str) -> pd.DataFrame:
    """Load (and merge, if needed) all fund files for a given category.
    Returns a wide dataframe with Date column and one column per fund.
    """
    files = CATEGORY_FILES[category]
    frames = [load_fund_file(os.path.join(data_dir, f)) for f in files]
    if len(frames) == 1:
        df = frames[0]
    else:
        df = frames[0]
        for nxt in frames[1:]:
            df = pd.merge(df, nxt, on="Date", how="outer")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_index(data_dir: str, index_name: str) -> pd.DataFrame:
    return load_index_file(os.path.join(data_dir, INDEX_FILES[index_name]))


@st.cache_data(show_spinner=False)
def get_fund_universe(data_dir: str) -> dict:
    """Return {fund_name: category} mapping across all categories."""
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
# Analysis helpers
# ----------------------------------------------------------------------------

def get_series(data_dir: str, name: str, kind: str) -> pd.DataFrame:
    """Return a 2-column DataFrame [Date, Value] for the given fund or index."""
    if kind == "index":
        idx = load_index(data_dir, name)
        return idx.rename(columns={"Close Price": "Value"})
    # else: fund — find its category
    universe = get_fund_universe(data_dir)
    cat = universe.get(name)
    if cat is None:
        return pd.DataFrame(columns=["Date", "Value"])
    df = load_category(data_dir, cat)
    out = df[["Date", name]].dropna().rename(columns={name: "Value"})
    return out.reset_index(drop=True)


def cagr(start_val: float, end_val: float, years: float) -> float:
    """Compound Annual Growth Rate as a decimal (0.12 = 12%)."""
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def value_near(series: pd.DataFrame, target_date: pd.Timestamp,
               tolerance_days: int = 7) -> tuple:
    """Pick the nearest available (Date, Value) at or before target_date.

    Falls back to the nearest date on either side within tolerance_days.
    Returns (None, None) if nothing close is available.
    """
    if series.empty:
        return None, None
    s = series.sort_values("Date").reset_index(drop=True)
    # Prefer the last value at or before target
    at_or_before = s[s["Date"] <= target_date]
    if not at_or_before.empty:
        row = at_or_before.iloc[-1]
        # Verify it's within tolerance — otherwise it might be way too old
        if (target_date - row["Date"]).days <= tolerance_days * 5:
            return row["Date"], row["Value"]
    # Else: pick the closest date overall within tolerance
    s["diff"] = (s["Date"] - target_date).abs()
    row = s.loc[s["diff"].idxmin()]
    if row["diff"].days <= tolerance_days:
        return row["Date"], row["Value"]
    return None, None


def period_return(series: pd.DataFrame, period_label: str,
                  as_of: pd.Timestamp) -> float:
    """Return for a standard period. Annualised (CAGR) if > 1 year,
    else absolute return (both as decimals)."""
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
    """Daily rolling CAGR over a `window_years` window. Returns df[Date, Return]."""
    if series.empty:
        return pd.DataFrame(columns=["Date", "Return"])
    s = series.sort_values("Date").set_index("Date").asfreq("D").ffill()
    window_days = int(round(window_years * 365.25))
    shifted = s["Value"].shift(window_days)
    cagr_series = (s["Value"] / shifted) ** (1 / window_years) - 1
    out = pd.DataFrame({"Date": s.index, "Return": cagr_series.values})
    return out.dropna().reset_index(drop=True)


def normalize_to_100(series: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    """Rebase a series to 100 on/after `start_date`."""
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
# Sidebar — data dir + navigation
# ----------------------------------------------------------------------------

st.sidebar.title("📈 MF Analysis")
data_dir = st.sidebar.text_input("Data folder", value=DEFAULT_DATA_DIR,
                                 help="Folder containing the .xlsx files")

if not os.path.isdir(data_dir):
    st.error(f"Data folder `{data_dir}` not found. "
             f"Place all the .xlsx files in this folder, or update the path in the sidebar.")
    st.stop()

page = st.sidebar.radio(
    "Analysis",
    ["1. Performance Graph",
     "2. Rolling Returns",
     "3. Category Leaderboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Returns are calculated from Adjusted NAVs. Periods > 1Y are shown as "
    "CAGR; ≤ 1Y are absolute returns. NAV gaps are forward-filled for "
    "rolling-return computations."
)

# Build a flat list of all funds with category tag (used by pages 1 & 2)
with st.spinner("Indexing fund universe..."):
    universe = get_fund_universe(data_dir)

# Pretty label "Fund Name (Category)" for selectors
fund_labels = sorted([f"{name} — {cat}" for name, cat in universe.items()])
label_to_fund = {f"{name} — {cat}": name for name, cat in universe.items()}
index_options = list(INDEX_FILES.keys())


# ----------------------------------------------------------------------------
# Page 1 — Performance graph
# ----------------------------------------------------------------------------

if page == "1. Performance Graph":
    st.title("Performance Comparison")
    st.caption("Pick any start date and compare funds/indices, all rebased to 100.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_labels = st.multiselect(
            "Select funds",
            options=fund_labels,
            default=fund_labels[:2] if len(fund_labels) >= 2 else fund_labels,
            help="Pick one or more funds to compare.",
        )
        selected_indices = st.multiselect(
            "Select indices",
            options=index_options,
            default=["NIFTY 50"],
        )
    with col2:
        # Determine valid date range
        today = date.today()
        default_start = today - timedelta(days=365 * 3)
        start_date = st.date_input("Start date", value=default_start,
                                   min_value=date(1990, 1, 1),
                                   max_value=today)
        end_date = st.date_input("End date", value=today,
                                 min_value=start_date,
                                 max_value=today)

    if not selected_labels and not selected_indices:
        st.info("Pick at least one fund or index above to plot.")
        st.stop()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Collect all series
    plot_rows = []
    summary_rows = []

    selected_funds = [label_to_fund[lbl] for lbl in selected_labels]
    items = [(f, "fund") for f in selected_funds] + \
            [(i, "index") for i in selected_indices]

    for name, kind in items:
        s = get_series(data_dir, name, kind)
        s = s[(s["Date"] >= start_ts) & (s["Date"] <= end_ts)]
        if s.empty:
            summary_rows.append({"Name": name, "Type": kind, "Note": "No data in range"})
            continue
        rebased = normalize_to_100(s, start_ts)
        rebased["Name"] = name
        rebased["Type"] = kind.capitalize()
        plot_rows.append(rebased)

        # Summary stats
        actual_start, start_val = value_near(s, start_ts)
        actual_end, end_val = value_near(s, end_ts)
        if start_val and end_val:
            yrs = (actual_end - actual_start).days / 365.25
            abs_ret = (end_val / start_val - 1) * 100
            ann_ret = (cagr(start_val, end_val, yrs) * 100) if yrs > 0 else np.nan
            summary_rows.append({
                "Name": name,
                "Type": kind.capitalize(),
                "From": actual_start.strftime("%Y-%m-%d"),
                "To": actual_end.strftime("%Y-%m-%d"),
                "Years": round(yrs, 2),
                "Total Return %": round(abs_ret, 2),
                "CAGR %": round(ann_ret, 2) if not np.isnan(ann_ret) else None,
            })

    if plot_rows:
        plot_df = pd.concat(plot_rows, ignore_index=True)
        fig = px.line(plot_df, x="Date", y="Value", color="Name",
                      line_dash="Type",
                      labels={"Value": "Rebased to 100"})
        fig.update_layout(height=520, hovermode="x unified",
                          legend_title="")
        fig.add_hline(y=100, line_dash="dot", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

        if summary_rows:
            st.subheader("Period summary")
            sdf = pd.DataFrame(summary_rows)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available in the selected date range.")


# ----------------------------------------------------------------------------
# Page 2 — Rolling returns
# ----------------------------------------------------------------------------

elif page == "2. Rolling Returns":
    st.title("Rolling Returns Comparison")
    st.caption("Daily rolling CAGR over a chosen window. Compare 1Y, 3Y, or 5Y rolling returns "
               "across 2–3 funds and an index.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_labels = st.multiselect(
            "Select up to 3 funds",
            options=fund_labels,
            default=fund_labels[:2] if len(fund_labels) >= 2 else fund_labels,
            max_selections=3,
        )
        selected_indices = st.multiselect(
            "Select indices",
            options=index_options,
            default=["NIFTY 50"],
        )
    with col2:
        window = st.radio("Rolling window",
                          options=[1, 3, 5],
                          index=1,
                          format_func=lambda x: f"{x}-Year")
        today = date.today()
        default_start = today - timedelta(days=365 * 10)
        start_date = st.date_input("Plot from", value=default_start,
                                   min_value=date(1990, 1, 1),
                                   max_value=today)

    if not selected_labels and not selected_indices:
        st.info("Pick at least one fund or index above.")
        st.stop()

    start_ts = pd.Timestamp(start_date)

    plot_rows = []
    stat_rows = []
    selected_funds = [label_to_fund[lbl] for lbl in selected_labels]
    items = [(f, "fund") for f in selected_funds] + \
            [(i, "index") for i in selected_indices]

    for name, kind in items:
        s = get_series(data_dir, name, kind)
        if s.empty:
            continue
        rr = rolling_returns(s, window)
        rr = rr[rr["Date"] >= start_ts]
        if rr.empty:
            stat_rows.append({"Name": name, "Note": "Not enough history for this window"})
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

    if plot_rows:
        plot_df = pd.concat(plot_rows, ignore_index=True)
        fig = px.line(plot_df, x="Date", y="Return %", color="Name",
                      line_dash="Type",
                      labels={"Return %": f"{window}Y Rolling CAGR (%)"})
        fig.update_layout(height=520, hovermode="x unified", legend_title="")
        fig.add_hline(y=0, line_dash="dot", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling-return statistics")
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        # Distribution view
        st.subheader("Distribution of rolling returns")
        fig2 = px.histogram(plot_df, x="Return %", color="Name",
                            barmode="overlay", opacity=0.55,
                            nbins=50)
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No rolling-return data could be computed for the current selections.")


# ----------------------------------------------------------------------------
# Page 3 — Category leaderboard
# ----------------------------------------------------------------------------

elif page == "3. Category Leaderboard":
    st.title("Category Leaderboard")
    st.caption("Rank every fund in a category by return for a chosen period. "
               "Periods > 1Y are annualised (CAGR).")

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", options=list(CATEGORY_FILES.keys()))
    with col2:
        period = st.selectbox("Period", options=list(PERIOD_DAYS.keys()), index=3)
    with col3:
        today = date.today()
        as_of = st.date_input("As of", value=today,
                              min_value=date(1995, 1, 1),
                              max_value=today)

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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Funds with data", len(df))
        c2.metric("Top return %", f"{df['Return %'].iloc[0]:.2f}")
        c3.metric("Benchmark %", f"{bench_pct:.2f}")
        c4.metric("Funds beating benchmark", f"{n_beat} / {len(df)}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Funds with data", len(df))
        c2.metric("Top return %", f"{df['Return %'].iloc[0]:.2f}")
        c3.metric("Median return %", f"{df['Return %'].median():.2f}")

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
        x=label,
        y="Fund",
        orientation="h",
        color=label,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=max(420, 24 * len(plot_df)),
                      yaxis_title="", coloraxis_showscale=False)
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
