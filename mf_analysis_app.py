"""
Index Rolling Returns Dashboard
===============================
Single-view Streamlit app for Indian equity indices:
  • a rolling-CAGR chart for whichever indices you select, and
  • one complete table of every index's median 1Y / 3Y / 5Y rolling return.

All index data is read from a single long-format file (indices.xlsx) with
columns: Index Name | Date | Close Price.

Run:
    streamlit run mf_analysis_app.py

Default: indices.xlsx sits next to this script.
"""

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Index Rolling Returns", page_icon="📈", layout="wide")

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
INDICES_FILE = "indices.xlsx"
ROLLING_WINDOWS = [1, 3, 5]  # years

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_indices(data_dir: str) -> dict:
    """Read the long-format indices file into {index_name: DataFrame[Date, Value]}.

    Layout: title row, blank row, header row (Index Name | Date | Close Price),
    then one row per index per date. Index names are discovered from the file,
    so adding a new index to the export needs no code change."""
    path = os.path.join(data_dir, INDICES_FILE)
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path, header=2)
    df = df.rename(columns={
        df.columns[0]: "Index Name",
        df.columns[1]: "Date",
        df.columns[2]: "Close Price",
    })
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close Price"] = pd.to_numeric(df["Close Price"], errors="coerce")
    df = df.dropna(subset=["Index Name", "Date", "Close Price"])

    out = {}
    for name, g in df.groupby("Index Name"):
        s = (g[["Date", "Close Price"]]
             .rename(columns={"Close Price": "Value"})
             .sort_values("Date")
             .drop_duplicates(subset="Date", keep="last")
             .reset_index(drop=True))
        out[str(name)] = s
    return out


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def rolling_returns(series: pd.DataFrame, window_years: int) -> pd.DataFrame:
    """Daily rolling CAGR over the given window. Price gaps forward-filled."""
    if series.empty:
        return pd.DataFrame(columns=["Date", "Return"])
    s = series.sort_values("Date").set_index("Date").asfreq("D").ffill()
    window_days = int(round(window_years * 365.25))
    shifted = s["Value"].shift(window_days)
    cagr_series = (s["Value"] / shifted) ** (1 / window_years) - 1
    return (pd.DataFrame({"Date": s.index, "Return": cagr_series.values})
            .dropna().reset_index(drop=True))


@st.cache_data(show_spinner=False)
def median_table(data_dir: str) -> pd.DataFrame:
    """Every index's median rolling return for each window, over full history."""
    idx = load_indices(data_dir)
    rows = []
    for name, s in idx.items():
        row = {"Index": name, "Data start": s["Date"].min().strftime("%Y-%m-%d")}
        for w in ROLLING_WINDOWS:
            rr = rolling_returns(s, w)
            row[f"Median {w}Y %"] = round(rr["Return"].median() * 100, 2) if not rr.empty else None
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = f"Median {ROLLING_WINDOWS[-1]}Y %"
    return df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Sidebar (settings + methodology only -- no navigation)
# ----------------------------------------------------------------------------

st.sidebar.markdown("# 📈 Index Rolling Returns")
st.sidebar.caption("Indian equity indices")
st.sidebar.markdown("---")
with st.sidebar.expander("Settings", expanded=False):
    data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR,
                             help=f"Folder containing {INDICES_FILE}")

if not os.path.isdir(data_dir):
    st.error(f"Data folder `{data_dir}` not found. Update the path in the sidebar settings.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Rolling returns are daily rolling CAGR over the chosen window. "
    "Price gaps are forward-filled. Medians in the table are taken over each "
    "index's full available history."
)

indices = load_indices(data_dir)
if not indices:
    st.error(f"`{INDICES_FILE}` not found in `{data_dir}`. Place the indices file there and reload.")
    st.stop()

index_options = list(indices.keys())


# ----------------------------------------------------------------------------
# Main -- Rolling Returns
# ----------------------------------------------------------------------------

st.title("Rolling Returns")
st.caption("Daily rolling CAGR for the selected indices, plus median rolling "
           "returns across all indices.")

sel_col, opt_col = st.columns([3, 2], gap="large")

with sel_col:
    selected = st.multiselect(
        "Indices", options=index_options, default=index_options,
        help="Choose which indices to plot on the chart below.",
    )

with opt_col:
    window = st.radio("Rolling window", options=ROLLING_WINDOWS, index=1,
                      format_func=lambda x: f"{x}-Year", horizontal=True)
    today = date.today()
    default_start = today - timedelta(days=365 * 10)
    start_date = st.date_input("Plot from", value=default_start,
                               min_value=date(1990, 1, 1),
                               max_value=today, key="plot_from")

# ---- Chart -----------------------------------------------------------------
start_ts = pd.Timestamp(start_date)

if not selected:
    st.info("Pick at least one index to plot.")
else:
    plot_rows = []
    for name in selected:
        rr = rolling_returns(indices[name], window)
        rr = rr[rr["Date"] >= start_ts]
        if rr.empty:
            continue
        rr = rr.copy()
        rr["Index"] = name
        rr["Return %"] = rr["Return"] * 100
        plot_rows.append(rr)

    if not plot_rows:
        st.warning("No rolling-return data in the selected range for these indices.")
    else:
        plot_df = pd.concat(plot_rows, ignore_index=True)
        fig = px.line(plot_df, x="Date", y="Return %", color="Index",
                      labels={"Return %": f"{window}Y Rolling CAGR (%)"})
        fig.update_layout(
            height=500, hovermode="x unified", legend_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        fig.add_hline(y=0, line_dash="dot", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ---- Median table (all indices) -------------------------------------------
st.subheader("Median rolling returns -- all indices")
st.caption("Median of each index's daily rolling CAGR, over its full history. "
           "Greener = higher within each column.")

table = median_table(data_dir)
median_cols = [f"Median {w}Y %" for w in ROLLING_WINDOWS]

styler = (
    table.style
    .format({c: "{:.2f}%" for c in median_cols}, na_rep="--")
    .background_gradient(cmap="RdYlGn", subset=median_cols)
    .set_properties(subset=median_cols, **{"font-weight": "600"})
)

st.dataframe(styler, use_container_width=True, hide_index=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button("Download table (CSV)", data=csv,
                   file_name="index_median_rolling_returns.csv", mime="text/csv")
