"""
Mutual Fund Analysis Dashboard
==============================
Streamlit app for analyzing Indian mutual funds across categories and comparing
them against equity indices.

Data sources (all sit next to this script by default):
  • Per-category fund files (wide layout, long history to ~Feb):
      largecap1.xlsx, largecap2.xlsx, largeandmidcapa.xlsx, midcap.xlsx,
      smallcap.xlsx, flexicap1.xlsx, flexicap2.xlsx, multicap.xlsx
  • Combined 1-year file for ALL funds (wide layout), extends/overlays the
      recent window and also contributes any fund missing from the base files:
      1yearfundsallcartegories.xlsx
  • Indices file (long layout: Index Name | Date | Close Price):
      indices.xlsx   (NIFTY 50, NIFTY 500, Nifty Midcap 150, Nifty Smallcap 250)

Every fund present in either fund source is included. Funds not found in a base
file are slotted into a category inferred from the scheme name.

Run:
    streamlit run mf_analysis_app.py
"""

import os
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

st.set_page_config(page_title="MF Analysis", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { font-size: 1.85rem !important; margin-bottom: 0.2rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.15rem !important; margin-top: 1.2rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.0rem !important; margin-top: 1rem !important; font-weight: 600 !important; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; }
        section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
        section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div { color: #111827 !important; }
        section[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; font-weight: 700 !important; color: #111827 !important; }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] * { color: #6b7280 !important; font-size: 0.82rem !important; line-height: 1.45 !important; }
        section[data-testid="stSidebar"] strong { color: #6b7280 !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.08em; }
        section[data-testid="stSidebar"] hr { margin: 1rem 0 0.6rem 0; border-color: #e5e7eb; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label { font-size: 0.95rem !important; font-weight: 500 !important; color: #1f2937 !important; padding: 0.4rem 0; cursor: pointer; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { color: #2563eb !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA_DIR = "."

# Base per-category files (long history). Category = which file a fund lives in.
CATEGORY_FILES = {
    "Large Cap":          ["largecap1.xlsx", "largecap2.xlsx"],
    "Large & Mid Cap":    ["largeandmidcapa.xlsx"],
    "Mid Cap":            ["midcap.xlsx"],
    "Small Cap":          ["smallcap.xlsx"],
    "Flexi Cap":          ["flexicap1.xlsx", "flexicap2.xlsx"],
    "Multi Cap":          ["multicap.xlsx"],
}

# Categories that exist only in the combined file (no base history file).
EXTRA_CATEGORIES = ["Equity Long-Short / SIF"]
ALL_CATEGORIES = list(CATEGORY_FILES.keys()) + EXTRA_CATEGORIES

# Combined 1-year, all-funds file. Overlays recent NAVs on the base files and
# supplies any fund the base files don't have.
UPDATE_FILE = "1yearfundsallcartegories.xlsx"

# Long-format indices file (all indices in one sheet).
INDICES_FILE = "indices.xlsx"

CATEGORY_DEFAULT_INDEX = {
    "Large Cap":                "NIFTY 50",
    "Large & Mid Cap":          "NIFTY 500",
    "Mid Cap":                  "Nifty Midcap 150",
    "Small Cap":                "Nifty Smallcap 250",
    "Flexi Cap":                "NIFTY 500",
    "Multi Cap":                "NIFTY 500",
    "Equity Long-Short / SIF":  "NIFTY 500",
}

PERIOD_DAYS = {
    "1M": 30, "3M": 91, "6M": 182, "1Y": 365,
    "3Y": 365 * 3, "5Y": 365 * 5, "10Y": 365 * 10,
}

# ----------------------------------------------------------------------------
# Category inference (for funds not present in any base file)
# ----------------------------------------------------------------------------

def infer_category(name: str) -> str:
    n = str(name).lower().replace("-", " ").replace("&", " and ")
    n = re.sub(r"\s+", " ", n)
    if "long short" in n or "ex top 100" in n:
        return "Equity Long-Short / SIF"
    if "large and mid" in n or ("large" in n and "mid" in n):
        return "Large & Mid Cap"
    if "flexi" in n:
        return "Flexi Cap"
    if "multi cap" in n or "multicap" in n:
        return "Multi Cap"
    if "large" in n:
        return "Large Cap"
    if "mid cap" in n or "midcap" in n:
        return "Mid Cap"
    if "small" in n:
        return "Small Cap"
    return "Multi Cap"  # diversified fallback (no funds currently hit this)

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
def load_update_file(data_dir: str) -> pd.DataFrame:
    """Combined 1-year, all-funds file (wide). Empty frame if absent."""
    path = os.path.join(data_dir, UPDATE_FILE)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date"])
    return load_fund_file(path)


@st.cache_data(show_spinner=False)
def load_indices(data_dir: str) -> dict:
    """Long-format indices file -> {index_name: DataFrame[Date, Value]}."""
    path = os.path.join(data_dir, INDICES_FILE)
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path, header=2)
    df = df.rename(columns={df.columns[0]: "Index Name",
                            df.columns[1]: "Date",
                            df.columns[2]: "Close Price"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close Price"] = pd.to_numeric(df["Close Price"], errors="coerce")
    df = df.dropna(subset=["Index Name", "Date", "Close Price"])
    out = {}
    for name, g in df.groupby("Index Name"):
        out[str(name)] = (g[["Date", "Close Price"]]
                          .rename(columns={"Close Price": "Value"})
                          .sort_values("Date")
                          .drop_duplicates(subset="Date", keep="last")
                          .reset_index(drop=True))
    return out


@st.cache_data(show_spinner=False)
def get_base_categories(data_dir: str) -> dict:
    """{fund: category} taken from the base files (authoritative)."""
    m = {}
    for cat, files in CATEGORY_FILES.items():
        for f in files:
            path = os.path.join(data_dir, f)
            if os.path.exists(path):
                df = load_fund_file(path)
                for c in df.columns:
                    if c != "Date":
                        m.setdefault(c, cat)
    return m


@st.cache_data(show_spinner=True)
def get_fund_universe(data_dir: str) -> dict:
    """{fund: category} across base files AND the combined file. Base files win;
    combined-only funds get a category inferred from the scheme name."""
    universe = dict(get_base_categories(data_dir))
    upd = load_update_file(data_dir)
    for c in upd.columns:
        if c != "Date" and c not in universe:
            universe[c] = infer_category(c)
    return universe


@st.cache_data(show_spinner=True)
def load_category(data_dir: str, category: str) -> pd.DataFrame:
    """All funds in a category (Date + one column per fund), assembled from base
    files and extended/added-to by the combined file. Overlay uses update() so
    older base data is never clobbered by a blank."""
    # 1. base files for this category (horizontal merge)
    frames = []
    for f in CATEGORY_FILES.get(category, []):
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            frames.append(load_fund_file(path))
    if frames:
        base = frames[0]
        for nxt in frames[1:]:
            base = pd.merge(base, nxt, on="Date", how="outer")
        base = base.sort_values("Date").reset_index(drop=True)
    else:
        base = pd.DataFrame(columns=["Date"])

    # 2. every fund the universe assigns to this category
    universe = get_fund_universe(data_dir)
    target = [f for f, c in universe.items() if c == category]
    if not target:
        return pd.DataFrame(columns=["Date"])

    # 3. start from base, ensure all target funds exist as columns
    if not base.empty:
        merged = base.set_index("Date")
    else:
        merged = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    for f in target:
        if f not in merged.columns:
            merged[f] = np.nan

    # 4. overlay / extend with the combined file
    upd = load_update_file(data_dir)
    if not upd.empty:
        cols = [f for f in target if f in upd.columns]
        if cols:
            add = upd.set_index("Date")[cols]
            merged = merged.reindex(merged.index.union(add.index))
            merged.update(add)

    keep = [f for f in target if f in merged.columns]
    return merged[keep].sort_index().reset_index()


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def get_index_series(data_dir: str, name: str) -> pd.DataFrame:
    return load_indices(data_dir).get(name, pd.DataFrame(columns=["Date", "Value"]))


def get_series(data_dir: str, name: str, kind: str) -> pd.DataFrame:
    if kind == "index":
        return get_index_series(data_dir, name)
    cat = get_fund_universe(data_dir).get(name)
    if cat is None:
        return pd.DataFrame(columns=["Date", "Value"])
    df = load_category(data_dir, cat)
    if name not in df.columns:
        return pd.DataFrame(columns=["Date", "Value"])
    return df[["Date", name]].dropna().rename(columns={name: "Value"}).reset_index(drop=True)


def cagr(start_val, end_val, years):
    if start_val is None or end_val is None or start_val <= 0 or end_val <= 0 or years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def value_near(series: pd.DataFrame, target_date: pd.Timestamp, tolerance_days: int = 7):
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

def fund_picker(label_prefix, universe, key_prefix, max_funds=None, default_count=2):
    cat_options = ["All categories"] + ALL_CATEGORIES
    cat = st.selectbox(f"{label_prefix} -- category filter",
                       options=cat_options, index=0, key=f"{key_prefix}_cat")
    if cat == "All categories":
        filtered = sorted(universe.keys())
    else:
        filtered = sorted([f for f, c in universe.items() if c == cat])
    if not filtered:
        st.info("No funds in this category.")
        return []
    default = filtered[:default_count] if len(filtered) >= default_count else filtered
    kwargs = dict(options=filtered, default=default, key=f"{key_prefix}_sel",
                  help="Type to search. Use the category filter above to narrow.")
    if max_funds is not None:
        kwargs["max_selections"] = max_funds
    return st.multiselect(label_prefix, **kwargs)


def latest_inception(data_dir, fund_names):
    starts = []
    for f in fund_names:
        s = get_series(data_dir, f, "fund")
        if not s.empty:
            starts.append(s["Date"].min())
    return max(starts) if starts else None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.markdown("# 📈 MF Analysis")
st.sidebar.caption("Indian mutual fund dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("Navigation",
                        ["Performance Graph", "Rolling Returns", "Category Returns"],
                        label_visibility="collapsed")

st.sidebar.markdown("---")
with st.sidebar.expander("Settings", expanded=False):
    data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR,
                             help="Folder containing the .xlsx files")

if not os.path.isdir(data_dir):
    st.error(f"Data folder `{data_dir}` not found. Update the path in the sidebar settings.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Returns from Adjusted NAVs. Periods > 1Y are CAGR; <= 1Y are absolute "
    "returns. NAV gaps are forward-filled for rolling computations. Indices come "
    "from indices.xlsx; recent fund NAVs from the combined 1-year file."
)

with st.spinner("Indexing fund universe..."):
    universe = get_fund_universe(data_dir)

indices = load_indices(data_dir)
index_options = list(indices.keys())
default_index = ["NIFTY 50"] if "NIFTY 50" in indices else (index_options[:1] if index_options else [])

# Confirmation of coverage
n_base = len(get_base_categories(data_dir))
n_total = len(universe)
st.sidebar.caption(f"Funds loaded: {n_total} "
                   f"({n_base} from base files, {n_total - n_base} added from the combined file). "
                   f"Indices: {len(index_options)}.")


# ============================================================================
# PAGE 1 -- Performance Graph
# ============================================================================

if page == "Performance Graph":
    st.title("Performance Comparison")
    st.caption("Compare funds and indices rebased to 100. The chart auto-starts "
               "from the most recent fund's inception so every line begins together.")

    sel_col, date_col = st.columns([3, 2], gap="large")
    with sel_col:
        selected_funds = fund_picker("Funds", universe, key_prefix="p1", default_count=2)
        selected_indices = st.multiselect("Indices", options=index_options,
                                          default=default_index, key="p1_idx")

    today = date.today()
    auto_start = latest_inception(data_dir, selected_funds)
    suggested_start = auto_start.date() if auto_start is not None else today - timedelta(days=365 * 3)

    with date_col:
        use_auto = st.toggle("Auto-start from most recent fund", value=True,
                             help="When on, the chart begins on the inception date of the latest fund selected.")
        if use_auto and auto_start is not None:
            start_date = suggested_start
            st.text_input("Start date", value=str(suggested_start), disabled=True, key="p1_sd_disp")
        else:
            start_date = st.date_input("Start date", value=suggested_start,
                                       min_value=date(1990, 1, 1), max_value=today, key="p1_sd")
        end_date = st.date_input("End date", value=today, min_value=start_date, max_value=today, key="p1_ed")

    if not selected_funds and not selected_indices:
        st.info("Pick at least one fund or index above.")
        st.stop()

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    items = [(f, "fund") for f in selected_funds] + [(i, "index") for i in selected_indices]

    plot_rows = []
    for name, kind in items:
        s_full = get_series(data_dir, name, kind)
        s = s_full[(s_full["Date"] >= start_ts) & (s_full["Date"] <= end_ts)]
        if s.empty:
            continue
        rebased = normalize_to_100(s, start_ts)
        rebased["Name"] = name
        rebased["Type"] = kind.capitalize()
        plot_rows.append(rebased)

    if not plot_rows:
        st.warning("No data in the selected range.")
        st.stop()

    plot_df = pd.concat(plot_rows, ignore_index=True)
    fig = px.line(plot_df, x="Date", y="Value", color="Name", line_dash="Type",
                  labels={"Value": "Rebased to 100"})
    fig.update_layout(height=500, hovermode="x unified", legend_title="",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25))
    fig.add_hline(y=100, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2 -- Rolling Returns
# ============================================================================

elif page == "Rolling Returns":
    st.title("Rolling Returns")
    st.caption("Daily rolling CAGR over a chosen window. Compare 1Y, 3Y, or 5Y "
               "rolling returns across funds and an index.")

    sel_col, opt_col = st.columns([3, 2], gap="large")
    with sel_col:
        selected_funds = fund_picker("Funds (up to 3)", universe, key_prefix="p2",
                                     max_funds=3, default_count=2)
        selected_indices = st.multiselect("Indices", options=index_options,
                                          default=default_index, key="p2_idx")
    with opt_col:
        window = st.radio("Rolling window", options=[1, 3, 5], index=1,
                          format_func=lambda x: f"{x}-Year", horizontal=True)
        today = date.today()
        default_start = today - timedelta(days=365 * 10)
        start_date = st.date_input("Plot from", value=default_start,
                                   min_value=date(1990, 1, 1), max_value=today, key="p2_sd")

    if not selected_funds and not selected_indices:
        st.info("Pick at least one fund or index above.")
        st.stop()

    start_ts = pd.Timestamp(start_date)
    items = [(f, "fund") for f in selected_funds] + [(i, "index") for i in selected_indices]

    plot_rows, stat_rows = [], []
    for name, kind in items:
        s = get_series(data_dir, name, kind)
        if s.empty:
            continue
        rr = rolling_returns(s, window)
        rr = rr[rr["Date"] >= start_ts]
        if rr.empty:
            stat_rows.append({"Name": name, "Type": kind.capitalize(), "Note": "Not enough history"})
            continue
        rr["Name"] = name
        rr["Type"] = kind.capitalize()
        rr["Return %"] = rr["Return"] * 100
        plot_rows.append(rr)
        stat_rows.append({
            "Name": name, "Type": kind.capitalize(),
            "Mean %": round(rr["Return %"].mean(), 2),
            "Median %": round(rr["Return %"].median(), 2),
            "Min %": round(rr["Return %"].min(), 2),
            "Max %": round(rr["Return %"].max(), 2),
            "Std %": round(rr["Return %"].std(), 2),
            "% > 0": round((rr["Return %"] > 0).mean() * 100, 1),
            "Obs": len(rr),
        })

    if not plot_rows:
        st.warning("No rolling-return data computed for the current selections.")
        st.stop()

    plot_df = pd.concat(plot_rows, ignore_index=True)
    fig = px.line(plot_df, x="Date", y="Return %", color="Name", line_dash="Type",
                  labels={"Return %": f"{window}Y Rolling CAGR (%)"})
    fig.update_layout(height=500, hovermode="x unified", legend_title="",
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25))
    fig.add_hline(y=0, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 3 -- Category Returns
# ============================================================================

elif page == "Category Returns":
    st.title("Category Returns")
    st.caption("Every fund in the chosen category, with returns across all standard "
               "periods. <= 1 year shows absolute return; > 1 year shows CAGR.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        category = st.selectbox("Category", options=ALL_CATEGORIES)
    with c2:
        today = date.today()
        as_of = st.date_input("As of", value=today, min_value=date(1995, 1, 1), max_value=today)
    with c3:
        sort_period = st.selectbox("Sort by", options=list(PERIOD_DAYS.keys()), index=4,
                                   help="Sort the table by this period's return (descending)")

    default_idx_name = CATEGORY_DEFAULT_INDEX.get(category, "NIFTY 500")
    show_benchmark = st.checkbox(f"Include benchmark ({default_idx_name})", value=True)

    as_of_ts = pd.Timestamp(as_of)
    cat_df = load_category(data_dir, category)
    funds = [c for c in cat_df.columns if c != "Date"]

    rows = []
    for fund in funds:
        s = cat_df[["Date", fund]].dropna().rename(columns={fund: "Value"})
        if s.empty:
            continue
        row = {"Fund": fund, "Data start": s["Date"].min().strftime("%Y-%m-%d")}
        any_data = False
        for plabel in PERIOD_DAYS:
            r = period_return(s, plabel, as_of_ts)
            if pd.notna(r):
                row[plabel] = round(r * 100, 2); any_data = True
            else:
                row[plabel] = None
        if any_data:
            rows.append(row)

    if not rows:
        st.warning("No funds in this category have enough data.")
        st.stop()

    df = pd.DataFrame(rows)

    benchmark_row = None
    if show_benchmark:
        idx = get_index_series(data_dir, default_idx_name)
        if not idx.empty:
            brow = {"Fund": f"* {default_idx_name} (benchmark)",
                    "Data start": idx["Date"].min().strftime("%Y-%m-%d")}
            for plabel in PERIOD_DAYS:
                r = period_return(idx, plabel, as_of_ts)
                brow[plabel] = round(r * 100, 2) if pd.notna(r) else None
            benchmark_row = brow

    df_sorted = df.sort_values(sort_period, ascending=False, na_position="last").reset_index(drop=True)

    if benchmark_row is not None and benchmark_row.get(sort_period) is not None:
        bench_val = benchmark_row[sort_period]
        n_beat = (df_sorted[sort_period] > bench_val).sum()
        n_total = df_sorted[sort_period].notna().sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Funds in category", len(df_sorted))
        m2.metric(f"Top {sort_period} return", f"{df_sorted[sort_period].iloc[0]:.2f}%")
        m3.metric(f"Benchmark {sort_period}", f"{bench_val:.2f}%")
        m4.metric(f"Beating benchmark ({sort_period})", f"{n_beat} / {n_total}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Funds in category", len(df_sorted))
        if df_sorted[sort_period].notna().any():
            m2.metric(f"Top {sort_period} return", f"{df_sorted[sort_period].max():.2f}%")
            m3.metric(f"Median {sort_period} return", f"{df_sorted[sort_period].median():.2f}%")

    if benchmark_row is not None:
        display_df = pd.concat([pd.DataFrame([benchmark_row]), df_sorted], ignore_index=True)
    else:
        display_df = df_sorted

    col_order = ["Fund", "Data start"] + list(PERIOD_DAYS.keys())
    display_df = display_df[col_order]

    column_config = {
        "Fund": st.column_config.TextColumn("Fund / Benchmark", width="large"),
        "Data start": st.column_config.TextColumn("Data start", width="small"),
    }
    for p in PERIOD_DAYS:
        vals = display_df[p].dropna()
        if vals.empty:
            column_config[p] = st.column_config.NumberColumn(p, format="%.2f%%")
        else:
            vmin, vmax = float(vals.min()), float(vals.max())
            pad = max(abs(vmax - vmin) * 0.05, 0.1)
            column_config[p] = st.column_config.ProgressColumn(
                p, format="%.2f%%", min_value=vmin - pad, max_value=vmax + pad)

    st.subheader(f"{category} -- returns as of {as_of_ts.date()}")
    st.caption(f"Sorted by {sort_period} (descending). "
               f"Bars are scaled within each column so the leader for each period is most filled.")

    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 column_config=column_config, height=min(600, 40 + 36 * len(display_df)))

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv,
                       file_name=f"{category}_returns_{as_of_ts.date()}.csv", mime="text/csv")
