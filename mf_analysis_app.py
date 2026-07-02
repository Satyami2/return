"""
Fund Rolling Returns Dashboard
==============================
Two tabs:
  1. All Funds  — a single table of every fund's median 1Y / 3Y / 5Y rolling return.
  2. Compare    — interactive rolling-return chart for selected funds vs indices.

Data sources (next to this script by default):
  • Per-category fund files (wide, long history):
      largecap1.xlsx, largecap2.xlsx, largeandmidcapa.xlsx, midcap.xlsx,
      smallcap.xlsx, flexicap1.xlsx, flexicap2.xlsx, multicap.xlsx
  • Combined 1-year all-funds file (wide): 1yearfundsallcartegories.xlsx
  • Indices file (long: Index Name | Date | Close Price): indices.xlsx

Every fund in either fund source is included; funds missing from the base files
are slotted into a category inferred from the scheme name.

Run:  streamlit run mf_analysis_app.py
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

st.set_page_config(page_title="Fund Rolling Returns", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { font-size: 1.85rem !important; margin-bottom: 0.2rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.15rem !important; margin-top: 1.2rem !important; font-weight: 600 !important; }
        section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
        section[data-testid="stSidebar"] * { color: #111827 !important; }
        section[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; font-weight: 700 !important; }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] * { color: #6b7280 !important; }
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
EXTRA_CATEGORIES = ["Equity Long-Short / SIF"]
ALL_CATEGORIES = list(CATEGORY_FILES.keys()) + EXTRA_CATEGORIES

UPDATE_FILE = "1yearfundsallcartegories.xlsx"
INDICES_FILE = "indices.xlsx"

ROLLING_WINDOWS = [1, 3, 5]  # years

# ----------------------------------------------------------------------------
# Category inference (funds not present in any base file)
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
    return "Multi Cap"

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
    path = os.path.join(data_dir, UPDATE_FILE)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date"])
    return load_fund_file(path)


@st.cache_data(show_spinner=False)
def load_indices(data_dir: str) -> dict:
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
    universe = dict(get_base_categories(data_dir))
    upd = load_update_file(data_dir)
    for c in upd.columns:
        if c != "Date" and c not in universe:
            universe[c] = infer_category(c)
    return universe


@st.cache_data(show_spinner=True)
def load_category(data_dir: str, category: str) -> pd.DataFrame:
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

    universe = get_fund_universe(data_dir)
    target = [f for f, c in universe.items() if c == category]
    if not target:
        return pd.DataFrame(columns=["Date"])

    if not base.empty:
        merged = base.set_index("Date")
    else:
        merged = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    for f in target:
        if f not in merged.columns:
            merged[f] = np.nan

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


def rolling_returns(series: pd.DataFrame, window_years: int) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["Date", "Return"])
    s = series.sort_values("Date").set_index("Date").asfreq("D").ffill()
    window_days = int(round(window_years * 365.25))
    shifted = s["Value"].shift(window_days)
    cagr_series = (s["Value"] / shifted) ** (1 / window_years) - 1
    return pd.DataFrame({"Date": s.index, "Return": cagr_series.values}).dropna().reset_index(drop=True)


@st.cache_data(show_spinner=True)
def fund_rolling_medians(data_dir: str) -> pd.DataFrame:
    """Median rolling 1Y/3Y/5Y return for every fund, over its full history."""
    rows = []
    for cat in ALL_CATEGORIES:
        df = load_category(data_dir, cat)
        for fund in [c for c in df.columns if c != "Date"]:
            s = df[["Date", fund]].dropna().rename(columns={fund: "Value"})
            if s.empty:
                continue
            row = {"Fund": fund, "Category": cat,
                   "Data start": s["Date"].min().strftime("%Y-%m-%d")}
            for w in ROLLING_WINDOWS:
                rr = rolling_returns(s, w)
                row[f"Median {w}Y %"] = round(rr["Return"].median() * 100, 2) if not rr.empty else None
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Median 3Y %", ascending=False, na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Sidebar (settings only)
# ----------------------------------------------------------------------------

st.sidebar.markdown("# 📈 Fund Rolling Returns")
st.sidebar.caption("Indian mutual funds")
st.sidebar.markdown("---")
with st.sidebar.expander("Settings", expanded=False):
    data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR,
                             help="Folder containing the .xlsx files")

if not os.path.isdir(data_dir):
    st.error(f"Data folder `{data_dir}` not found. Update the path in the sidebar settings.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Rolling returns are daily rolling CAGR over the chosen window. NAV gaps are "
    "forward-filled. Medians in the All Funds tab are over each fund's full history."
)

with st.spinner("Indexing fund universe..."):
    universe = get_fund_universe(data_dir)

indices = load_indices(data_dir)
index_options = list(indices.keys())
default_index = ["NIFTY 50"] if "NIFTY 50" in indices else (index_options[:1] if index_options else [])

n_base = len(get_base_categories(data_dir))
n_total = len(universe)
st.sidebar.caption(f"Funds loaded: {n_total} ({n_base} base + {n_total - n_base} combined). "
                   f"Indices: {len(index_options)}.")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_all, tab_cmp = st.tabs(["All Funds — Rolling Returns", "Compare — Funds vs Indices"])


# ===== TAB 1 — All funds rolling returns table ==============================
with tab_all:
    st.subheader("Rolling returns — all funds")
    st.caption("Median of each fund's daily rolling CAGR (1Y, 3Y, 5Y) over its full "
               "history. Greener = higher within each column. Blank = not enough history.")

    med = fund_rolling_medians(data_dir)
    if med.empty:
        st.warning("No fund data found.")
    else:
        cat_filter = st.selectbox("Category filter",
                                  options=["All categories"] + ALL_CATEGORIES, key="t1cat")
        view = med if cat_filter == "All categories" else med[med["Category"] == cat_filter]
        view = view.reset_index(drop=True)

        mcols = [f"Median {w}Y %" for w in ROLLING_WINDOWS]
        styler = (view.style
                  .format({c: "{:.2f}%" for c in mcols}, na_rep="--")
                  .background_gradient(cmap="RdYlGn", subset=mcols)
                  .set_properties(subset=mcols, **{"font-weight": "600"}))

        st.dataframe(styler, use_container_width=True, hide_index=True,
                     height=min(720, 44 + 34 * len(view)))

        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download table (CSV)", data=csv,
                           file_name="fund_median_rolling_returns.csv", mime="text/csv")


# ===== TAB 2 — Funds vs indices comparison ==================================
with tab_cmp:
    st.subheader("Rolling returns — funds vs indices")
    st.caption("Daily rolling CAGR over the chosen window for the selected funds "
               "and indices.")

    sel_col, opt_col = st.columns([3, 2], gap="large")
    with sel_col:
        cat = st.selectbox("Fund category filter",
                           options=["All categories"] + ALL_CATEGORIES, key="c_cat")
        if cat == "All categories":
            fund_opts = sorted(universe.keys())
        else:
            fund_opts = sorted([f for f, c in universe.items() if c == cat])
        default_funds = fund_opts[:2] if len(fund_opts) >= 2 else fund_opts
        selected_funds = st.multiselect("Funds (up to 5)", options=fund_opts,
                                        default=default_funds, max_selections=5,
                                        key="c_funds", help="Type to search.")
        selected_indices = st.multiselect("Indices (for comparison)", options=index_options,
                                          default=default_index, key="c_idx")
    with opt_col:
        window = st.radio("Rolling window", options=ROLLING_WINDOWS, index=1,
                          format_func=lambda x: f"{x}-Year", horizontal=True, key="c_win")
        today = date.today()
        start_date = st.date_input("Plot from", value=today - timedelta(days=365 * 10),
                                   min_value=date(1990, 1, 1), max_value=today, key="c_sd")

    if not selected_funds and not selected_indices:
        st.info("Pick at least one fund or index above.")
    else:
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
            rr = rr.copy()
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
            st.warning("No rolling-return data for the current selections in this range.")
        else:
            plot_df = pd.concat(plot_rows, ignore_index=True)
            fig = px.line(plot_df, x="Date", y="Return %", color="Name", line_dash="Type",
                          labels={"Return %": f"{window}Y Rolling CAGR (%)"})
            fig.update_layout(height=520, hovermode="x unified", legend_title="",
                              margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            fig.add_hline(y=0, line_dash="dot", opacity=0.4)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Statistics")
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
