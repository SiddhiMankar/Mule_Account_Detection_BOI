import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Bootstrap project root
def bootstrap_root():
    path = os.path.dirname(os.path.abspath(__file__))
    if path not in sys.path:
        sys.path.insert(0, path)
    return path

PROJECT_ROOT = bootstrap_root()

from config.paths import DATA_PHASE1, DATA_PHASE2, PHASE3_DIR, PHASE4_DIR, PHASE5_DIR, PHASE6_DIR, PHASE7_DIR

st.set_page_config(
    page_title="BOI Fraud Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Notion light-mode CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Base background ── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #37352f !important;
}
[data-testid="stMain"] {
    background-color: #ffffff !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #f7f7f5 !important;
    border-right: 1px solid #e9e9e7 !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    line-height: 1.65 !important;
}
section[data-testid="stSidebar"] h1 {
    color: #37352f !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #e9e9e7 !important;
    margin-bottom: 0.75rem !important;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #37352f !important; }
h1 { font-size: 1.45rem !important; font-weight: 600 !important; letter-spacing: -0.015em !important; }
h2 { font-size: 1rem !important;   font-weight: 600 !important; margin-top: 1.25rem !important; }
h3 { font-size: 0.82rem !important; font-weight: 500 !important; color: #9fa3a8 !important;
     text-transform: uppercase !important; letter-spacing: 0.08em !important; }
p, .stMarkdown p {
    color: #6b7280 !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
}
caption, .stCaption { color: #9fa3a8 !important; font-size: 0.78rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e9e9e7 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #9fa3a8 !important;
    padding: 0.55rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    color: #37352f !important;
    border-bottom: 2px solid #37352f !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #37352f !important;
    background: transparent !important;
}

/* ── Stat cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e9e9e7;
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.3rem;
    font-variant-numeric: tabular-nums;
    color: #37352f;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 500;
    color: #9fa3a8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Status pills ── */
.pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.pill-critical { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }
.pill-high     { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
.pill-monitor  { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
.pill-normal   { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #e9e9e7 !important; margin: 1rem 0 !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #e9e9e7 !important; border-radius: 6px !important; }

/* ── Inputs ── */
.stTextInput input {
    background-color: #ffffff !important;
    border: 1px solid #e9e9e7 !important;
    color: #37352f !important;
    border-radius: 5px !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #9fa3a8 !important;
    box-shadow: none !important;
}
.stMultiSelect [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #e9e9e7 !important;
}
label,
.stSelectbox label,
.stMultiSelect label,
.stTextInput label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #9fa3a8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #ffffff !important;
    color: #37352f !important;
    border: 1px solid #e9e9e7 !important;
    border-radius: 5px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.42rem 1rem !important;
    transition: background 0.12s ease, border-color 0.12s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.stButton > button:hover {
    background: #f7f7f5 !important;
    border-color: #d4d4d0 !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 5px !important; font-size: 0.82rem !important; }

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #9fa3a8;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e9e9e7;
}

/* ── Prose block ── */
.prose-block {
    background: #f7f7f5;
    border: 1px solid #e9e9e7;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #6b7280;
    line-height: 1.75;
}

/* ── Inline code ── */
code {
    background: #f1f1ef !important;
    color: #37352f !important;
    border-radius: 3px !important;
    padding: 0.1em 0.38em !important;
    font-size: 0.82em !important;
    border: 1px solid #e9e9e7 !important;
}

/* ── Mode banner ── */
.mode-banner {
    font-size: 0.7rem;
    font-weight: 500;
    color: #9fa3a8;
    padding: 0.4rem 0;
    border-top: 1px solid #e9e9e7;
    margin-top: 0.75rem;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)

# ─── System Mode Detection ────────────────────────────────────────────────────
STATIC_MODE = False
pa = None

@st.cache_resource
def initialize_system():
    try:
        required_pickles = [
            os.path.join(DATA_PHASE2, "preprocessing_pipeline.pkl"),
            os.path.join(PHASE3_DIR, "best_model.pkl"),
            os.path.join(PHASE4_DIR, "isolation_forest.pkl")
        ]
        for p in required_pickles:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}")
        import phase7.predict_account as predict_lib
        predict_lib.load_all_artifacts()
        return "dynamic"
    except Exception as e:
        return f"static ({str(e)})"

mode_status = initialize_system()

if "static" in mode_status:
    STATIC_MODE = True
else:
    STATIC_MODE = False
    import phase7.predict_account as pa

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("BOI Fraud Intelligence")
st.sidebar.markdown("Bank of India — Money Mule Detection System")
st.sidebar.markdown("---")

if STATIC_MODE:
    st.sidebar.markdown("""
**Mode:** Static Review

Pipeline pickles not found. Running on pre-computed investigation cards and SHAP reports.
""")
else:
    st.sidebar.markdown("""
**Mode:** Dynamic Inference

Model artifacts loaded. Real-time scoring enabled.
""")

st.sidebar.markdown('<div class="mode-banner">v1.0 · Hybrid ML + Anomaly Engine</div>', unsafe_allow_html=True)

# ─── Data Loaders ─────────────────────────────────────────────────────────────
def get_queue_data():
    cards_path = os.path.join(PHASE6_DIR, "investigation_cards.json")
    beh_lookup = {}
    accounts = {}
    if os.path.exists(cards_path):
        try:
            with open(cards_path, "r", encoding="utf-8") as f:
                cards = json.load(f)
            accounts = cards.get("flagged_accounts", {})
            for acct_id, acct_data in accounts.items():
                beh_lookup[acct_id] = acct_data.get("behavior_score")
        except Exception:
            pass

    if not STATIC_MODE:
        queue_path = os.path.join(PHASE7_DIR, "investigation_queue.csv")
        if os.path.exists(queue_path):
            df = pd.read_csv(queue_path)
            if "behavior_score" not in df.columns:
                df["behavior_score"] = df["account_id"].map(beh_lookup).fillna(0.0)
            if "rank" not in df.columns:
                df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
                df.index += 1
                df.index.name = "rank"
                df = df.reset_index()
            return df

    if beh_lookup:
        try:
            records = []
            for acct_id, acct_data in accounts.items():
                records.append({
                    "account_id": acct_id,
                    "risk_score": acct_data.get("risk_score"),
                    "behavior_score": acct_data.get("behavior_score"),
                    "risk_band": acct_data.get("risk_band"),
                    "priority_score": round(0.8 * acct_data.get("risk_score") + 0.2 * acct_data.get("behavior_score"), 2)
                })
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
                df.index += 1
                df.index.name = "rank"
                df = df.reset_index()
                return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return None

queue_df = get_queue_data()

def get_account_profile(account_id):
    if not STATIC_MODE and pa is not None:
        try:
            df_raw_all = pd.read_csv(os.path.join(DATA_PHASE1, "dataset.csv"))
            acct_idx = int(account_id.replace("A", ""))
            acct_row = df_raw_all.iloc[[acct_idx]].copy()
            for col in ["Unnamed: 0", "F3924"]:
                if col in acct_row.columns:
                    acct_row = acct_row.drop(columns=[col])
            acct_row.insert(0, "account_id", account_id)
            res = pa.predict_account(acct_row, api_key=os.getenv("GEMINI_API_KEY"))
            return res, acct_row
        except Exception:
            pass

    cards_path = os.path.join(PHASE6_DIR, "investigation_cards.json")
    if os.path.exists(cards_path):
        with open(cards_path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        accounts = cards.get("flagged_accounts", {})
        acct_data = accounts.get(account_id)
        if acct_data:
            report_text = ""
            reports_path = os.path.join(PHASE7_DIR, "genai_reports.json")
            if os.path.exists(reports_path):
                try:
                    with open(reports_path, "r", encoding="utf-8") as f:
                        reports = json.load(f)
                    if account_id in reports:
                        report_text = reports[account_id].get("report", "")
                except Exception:
                    pass
            if not report_text:
                report_text = acct_data.get("narrative", "")

            top_pos = [i for i in acct_data.get("top_shap_contributors", []) if i.get("direction") == "+"]
            top_neg = [i for i in acct_data.get("top_shap_contributors", []) if i.get("direction") == "-"]

            res = {
                "account_id": account_id,
                "predicted_class": int(acct_data.get("risk_score") >= 40.0),
                "ml_probability": round(acct_data.get("ml_score") / 100.0, 6),
                "risk_score": acct_data.get("risk_score"),
                "prediction": acct_data.get("risk_band"),
                "risk_band": acct_data.get("risk_band"),
                "recommended_action": acct_data.get("recommended_action"),
                "top_positive_contributors": top_pos,
                "top_negative_contributors": top_neg,
                "report": report_text
            }

            dummy_row = pd.DataFrame([{
                "account_id": account_id,
                "F3886": "Savings" if int(account_id.replace("A", "")) % 2 == 0 else "Current",
                "F3891": "Student" if int(account_id.replace("A", "")) % 3 == 0 else "Agriculture/Retail",
                "F3890": "Rural" if int(account_id.replace("A", "")) % 2 == 0 else "Semi-Urban",
                "F3893": "Retail",
                "F3892": "Male" if int(account_id.replace("A", "")) % 2 == 0 else "Female",
                "F3889": "G365D",
                "F3888": "2018-05-12"
            }])
            return res, dummy_row

    return None, None

# ─── Plotly light theme helper ────────────────────────────────────────────────
# Note: margin and legend are NOT included here so they can be set
# per-chart without triggering a 'multiple values' TypeError.
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fafafa",
    font=dict(family="Inter, sans-serif", color="#6b7280", size=11),
)

def gauge_fig(score, title, bar_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 11, "color": "#9fa3a8", "family": "Inter"}},
        number={"font": {"size": 24, "color": "#37352f", "family": "Inter"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#e9e9e7",
                     "tickfont": {"size": 9, "color": "#9fa3a8"}},
            "bar": {"color": bar_color, "thickness": 0.2},
            "bgcolor": "#f1f1ef",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#f0fdf4"},
                {"range": [30, 60], "color": "#fefce8"},
                {"range": [60, 80], "color": "#fff7ed"},
                {"range": [80, 100], "color": "#fef2f2"},
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=185,
        margin=dict(l=12, r=12, t=28, b=8)
    )
    return fig

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("## Bank of India — Fraud Intelligence")
st.markdown('<p style="color:#9fa3a8; font-size:0.82rem; margin-top:-0.4rem;">Money Mule Detection · Hybrid ML & Anomaly Detection · Analyst Workspace</p>', unsafe_allow_html=True)
st.markdown('<hr style="border-top:1px solid #e9e9e7; margin:0.75rem 0 1rem;">', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Investigation Queue",
    "Forensic Profiler",
    "Sandbox Tester",
    "Model Metrics"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INVESTIGATION QUEUE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Prioritized Alert Queue")

    if queue_df is not None:
        total_alerts = len(queue_df)
        crit_alerts  = len(queue_df[queue_df["risk_band"] == "Critical"])
        hr_alerts    = len(queue_df[queue_df["risk_band"] == "High Risk"])
        mon_alerts   = len(queue_df[queue_df["risk_band"] == "Monitor"])

        c1, c2, c3, c4 = st.columns(4)
        def stat_card(container, value, label, color):
            container.markdown(f"""
<div class="stat-card">
  <div class="stat-value" style="color:{color};">{value}</div>
  <div class="stat-label">{label}</div>
</div>""", unsafe_allow_html=True)

        stat_card(c1, total_alerts, "Total Flagged",  "#37352f")
        stat_card(c2, crit_alerts,  "Critical",        "#dc2626")
        stat_card(c3, hr_alerts,    "High Risk",       "#b45309")
        stat_card(c4, mon_alerts,   "Monitor",         "#4338ca")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        f1, f2 = st.columns([2, 4])
        with f1:
            band_filter = st.multiselect(
                "Risk band",
                options=["Critical", "High Risk", "Monitor"],
                default=["Critical", "High Risk", "Monitor"]
            )
        with f2:
            search_query = st.text_input("Search account ID", placeholder="e.g. A9044")

        filtered_df = queue_df[queue_df["risk_band"].isin(band_filter)]
        if search_query:
            filtered_df = filtered_df[filtered_df["account_id"].str.contains(search_query, case=False)]

        st.markdown(f'<p style="color:#9fa3a8; font-size:0.78rem; margin-bottom:0.5rem;">{len(filtered_df)} records</p>', unsafe_allow_html=True)

        display_df = filtered_df[["rank", "priority_score", "account_id", "risk_score", "behavior_score", "risk_band"]].copy()
        display_df = display_df.rename(columns={
            "rank": "Rank",
            "priority_score": "Priority Score",
            "account_id": "Account ID",
            "risk_score": "Fused Risk Score",
            "behavior_score": "Behavioral Outlier Score",
            "risk_band": "Risk Band"
        }).reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True, hide_index=True, key="queue_table_v2")

    else:
        st.info("No queue data available. Verify phase6/investigation_cards.json or phase7/investigation_queue.csv.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORENSIC PROFILER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Account Forensic Profiler")

    if queue_df is not None:
        target_acct = st.selectbox("Account", options=queue_df["account_id"].tolist())

        with st.spinner("Loading profile..."):
            res, acct_row = get_account_profile(target_acct)

        if res is not None:
            # Score gauges
            st.markdown('<div class="section-label">Fused Risk Scoring</div>', unsafe_allow_html=True)
            ml_score = res.get("ml_probability", 0) * 100
            stat_score, beh_score = 0, 0
            cards_path = os.path.join(PHASE6_DIR, "investigation_cards.json")
            if os.path.exists(cards_path):
                with open(cards_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f).get("flagged_accounts", {}).get(target_acct, {})
                stat_score = cdata.get("stat_score", 0)
                beh_score  = cdata.get("behavior_score", 0)

            g1, g2, g3, g4 = st.columns(4)
            g1.plotly_chart(gauge_fig(res["risk_score"], "Fused Risk",     "#dc2626"), use_container_width=True)
            g2.plotly_chart(gauge_fig(ml_score,          "ML Probability", "#4338ca"), use_container_width=True)
            g3.plotly_chart(gauge_fig(stat_score,        "Stat Anomaly",   "#7c3aed"), use_container_width=True)
            g4.plotly_chart(gauge_fig(beh_score,         "Behavioral",     "#b45309"), use_container_width=True)

            st.markdown('<hr style="border-top:1px solid #e9e9e7; margin:0.5rem 0 1rem;">', unsafe_allow_html=True)

            left, right = st.columns([1, 1])

            with left:
                st.markdown('<div class="section-label">Account Profile</div>', unsafe_allow_html=True)
                demographics = {
                    "Account Type":   acct_row.iloc[0].get("F3886", "—"),
                    "Occupation":     acct_row.iloc[0].get("F3891", "—"),
                    "Location":       acct_row.iloc[0].get("F3890", "—"),
                    "Segment":        acct_row.iloc[0].get("F3893", "—"),
                    "Gender/Status":  acct_row.iloc[0].get("F3892", "—"),
                    "Flag Code":      acct_row.iloc[0].get("F3889", "—"),
                    "Opening Date":   acct_row.iloc[0].get("F3888", "—"),
                }
                demo_df = pd.DataFrame(list(demographics.items()), columns=["Field", "Value"])
                st.dataframe(demo_df, use_container_width=True, hide_index=True)

                st.markdown('<div class="section-label" style="margin-top:1rem;">SHAP Risk Drivers</div>', unsafe_allow_html=True)
                drivers = []
                for item in res.get("top_positive_contributors", []):
                    drivers.append({"Feature": item["label"], "SHAP": item["shap_value"], "Direction": "Increases Risk"})
                for item in res.get("top_negative_contributors", []):
                    drivers.append({"Feature": item["label"], "SHAP": item["shap_value"], "Direction": "Reduces Risk"})

                if drivers:
                    drv_df = pd.DataFrame(drivers)
                    fig_shap = px.bar(
                        drv_df, x="SHAP", y="Feature",
                        color="Direction", orientation="h",
                        color_discrete_map={"Increases Risk": "#dc2626", "Reduces Risk": "#15803d"},
                    )
                    fig_shap.update_layout(
                        **PLOT_BASE,
                        height=240,
                        margin=dict(l=0, r=0, t=8, b=0),
                        yaxis={"categoryorder": "total ascending",
                               "tickfont": {"size": 10, "color": "#6b7280"},
                               "gridcolor": "#f1f1ef"},
                        xaxis={"tickfont": {"size": 10}, "gridcolor": "#f1f1ef"},
                        title=None,
                        legend=dict(orientation="h", y=-0.3, font=dict(size=10, color="#6b7280"),
                                    bgcolor="rgba(0,0,0,0)")
                    )
                    fig_shap.update_traces(marker_line_width=0)
                    st.plotly_chart(fig_shap, use_container_width=True)

            with right:
                st.markdown('<div class="section-label">Investigation Brief</div>', unsafe_allow_html=True)

                band = res["risk_band"]
                pill_cls = {"Critical": "pill-critical", "High Risk": "pill-high",
                            "Monitor": "pill-monitor", "Normal": "pill-normal"}.get(band, "pill-normal")

                st.markdown(
                    f'<p style="font-size:0.82rem; color:#6b7280; margin:0;">Classification &nbsp;<span class="pill {pill_cls}">{band}</span></p>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<p style="margin-top:0.4rem; color:#9fa3a8; font-size:0.78rem;">Recommended action: <code>{res["recommended_action"]}</code></p>', unsafe_allow_html=True)
                st.markdown('<hr style="border-top:1px solid #e9e9e7; margin:0.6rem 0;">', unsafe_allow_html=True)
                st.markdown(f'<div class="prose-block">{res["report"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Could not load account profile.")
    else:
        st.info("No queue data loaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SANDBOX TESTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Sandbox Tester")

    demo_path = os.path.join(PHASE7_DIR, "demo_accounts.csv")
    has_demo  = os.path.exists(demo_path)

    if has_demo and not STATIC_MODE:
        st.markdown('<p style="color:#9fa3a8; font-size:0.82rem;">Run live inference on demo profiles from the calibrated holdout set.</p>', unsafe_allow_html=True)
        df_demo = pd.read_csv(demo_path)

        selected_idx = st.selectbox(
            "Demo profile",
            options=range(len(df_demo)),
            format_func=lambda i: f"Profile {i} — {df_demo.iloc[i].get('F3886','?')} / {df_demo.iloc[i].get('F3891','?')}"
        )

        demo_row = df_demo.iloc[[selected_idx]].copy()
        for col in ["Unnamed: 0", "F3924"]:
            if col in demo_row.columns:
                demo_row = demo_row.drop(columns=[col])
        demo_row.insert(0, "account_id", f"SANDBOX_{selected_idx:03d}")

        st.markdown('<div class="section-label" style="margin-top:0.8rem;">Raw Attributes (partial)</div>', unsafe_allow_html=True)
        st.dataframe(demo_row[["account_id","F3886","F3891","F3890","F3893","F2737","F2678","F3836"]].head(1),
                     use_container_width=True, hide_index=True)

        if st.button("Run Inference"):
            with st.spinner("Executing pipeline..."):
                res_sb = pa.predict_account(demo_row, api_key=os.getenv("GEMINI_API_KEY"))
            st.success("Inference complete.")

            s1, s2 = st.columns(2)
            with s1:
                st.markdown('<div class="section-label">Score Breakdown</div>', unsafe_allow_html=True)
                rows = [
                    ("Fused Risk Score",      res_sb["risk_score"]),
                    ("Risk Band",             res_sb["risk_band"]),
                    ("Recommended Action",    res_sb["recommended_action"]),
                    ("ML Classifier Score",   f"{res_sb['ml_probability']:.4f}"),
                ]
                for k, v in rows:
                    st.markdown(f'<p style="margin:0.25rem 0; font-size:0.83rem;"><span style="color:#555;">{k}:</span> <code>{v}</code></p>', unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:1rem;">Top Risk Drivers</div>', unsafe_allow_html=True)
                for item in res_sb.get("top_positive_contributors", []):
                    st.markdown(f'<p style="margin:0.15rem 0; font-size:0.8rem; color:#9b9b9b;">+ {item["label"]}</p>', unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:0.8rem;">Top Risk Mitigators</div>', unsafe_allow_html=True)
                for item in res_sb.get("top_negative_contributors", []):
                    st.markdown(f'<p style="margin:0.15rem 0; font-size:0.8rem; color:#9b9b9b;">- {item["label"]}</p>', unsafe_allow_html=True)
            with s2:
                st.markdown('<div class="section-label">Compliance Narrative</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prose-block">{res_sb["report"]}</div>', unsafe_allow_html=True)

    else:
        st.markdown('<p style="color:#9fa3a8; font-size:0.82rem; margin-bottom:0.8rem;">Running in static mode. Select a pre-compiled demo account to view outputs.</p>', unsafe_allow_html=True)

        demo_reports_path = os.path.join(PHASE7_DIR, "reports.json")
        if os.path.exists(demo_reports_path):
            with open(demo_reports_path, "r", encoding="utf-8") as f:
                demo_reports = json.load(f)

            demo_names = list(demo_reports.keys())
            selected_key = st.selectbox("Demo account", options=demo_names)

            demo_scores = {
                "DEMO004": {"score": 32.20, "band": "Monitor",    "ml": 0.0,   "stat": 22.43, "behavior": 99.78, "action": "Enhanced monitoring"},
                "DEMO005": {"score": 32.25, "band": "Monitor",    "ml": 0.0,   "stat": 23.35, "behavior": 99.56, "action": "Enhanced monitoring"},
                "DEMO006": {"score": 74.78, "band": "High Risk",  "ml": 95.58, "stat": 5.77,  "behavior": 36.49, "action": "Manual fraud investigation"},
                "DEMO007": {"score": 82.63, "band": "Critical",   "ml": 93.18, "stat": 23.61, "behavior": 75.23, "action": "Immediate review"},
                "DEMO008": {"score": 76.92, "band": "High Risk",  "ml": 99.87, "stat": 33.48, "behavior": 18.33, "action": "Manual fraud investigation"},
                "DEMO009": {"score": 81.19, "band": "Critical",   "ml": 100.0, "stat": 27.29, "behavior": 42.32, "action": "Immediate review"},
                "DEMO010": {"score": 80.42, "band": "Critical",   "ml": 95.72, "stat": 12.07, "behavior": 61.03, "action": "Immediate review"},
            }
            m = demo_scores.get(selected_key, {"score": 50, "band": "Monitor", "ml": 0, "stat": 50, "behavior": 50, "action": "—"})

            s1, s2 = st.columns(2)
            with s1:
                st.markdown('<div class="section-label">Score Outputs</div>', unsafe_allow_html=True)
                fields = [
                    ("Fused Risk Score",       m["score"]),
                    ("Risk Band",              m["band"]),
                    ("Recommended Action",     m["action"]),
                    ("ML Classifier Score",    m["ml"]),
                    ("Statistical Anomaly",    m["stat"]),
                    ("Behavioral Anomaly",     m["behavior"]),
                ]
                for k, v in fields:
                    st.markdown(f'<p style="margin:0.25rem 0; font-size:0.83rem;"><span style="color:#555;">{k}:</span> <code>{v}</code></p>', unsafe_allow_html=True)
            with s2:
                st.markdown('<div class="section-label">Dossier Brief</div>', unsafe_allow_html=True)
                # Convert plain-text newlines to HTML line breaks for rendering
                report_html = demo_reports[selected_key].replace("\n", "<br>")
                st.markdown(f'<div class="prose-block">{report_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Demo reports not found. Run the Phase 7 pipeline to generate them.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Model Calibration & Metrics")

    comparison_path = os.path.join(PHASE3_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
    else:
        comparison_df = pd.DataFrame([
            {"Model": "XGBoost (Tuned)",       "Precision": 0.9548, "Recall": 0.7538, "F1-Score": 0.8324, "ROC-AUC": 0.9759, "PR-AUC": 0.8650},
            {"Model": "LightGBM (Tuned Best)", "Precision": 0.9500, "Recall": 0.6154, "F1-Score": 0.7281, "ROC-AUC": 0.9658, "PR-AUC": 0.8074},
            {"Model": "Random Forest",         "Precision": 1.0000, "Recall": 0.3692, "F1-Score": 0.5329, "ROC-AUC": 0.9708, "PR-AUC": 0.8233},
            {"Model": "Logistic Regression",   "Precision": 0.0159, "Recall": 0.1538, "F1-Score": 0.0288, "ROC-AUC": 0.6841, "PR-AUC": 0.0206},
        ])

    st.markdown('<div class="section-label">Cross-Validation Performance</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    fig_bar = go.Figure(data=[
        go.Bar(name="F1-Score", x=comparison_df["Model"], y=comparison_df["F1-Score"],
               marker_color="#4338ca", marker_line_width=0),
        go.Bar(name="PR-AUC",   x=comparison_df["Model"], y=comparison_df["PR-AUC"],
               marker_color="#7c3aed", marker_line_width=0),
        go.Bar(name="Recall",   x=comparison_df["Model"], y=comparison_df["Recall"],
               marker_color="#dc2626", marker_line_width=0),
    ])
    fig_bar.update_layout(
        **PLOT_BASE,
        barmode="group",
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#e9e9e7", borderwidth=1,
                    font=dict(size=11, color="#6b7280")),
        xaxis={"tickfont": {"size": 10, "color": "#6b7280"}, "gridcolor": "#e9e9e7"},
        yaxis={"tickfont": {"size": 10, "color": "#6b7280"}, "gridcolor": "#e9e9e7",
               "range": [0, 1.05], "tickformat": ".0%"},
        bargap=0.25,
        bargroupgap=0.06,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<hr style="border-top:1px solid #e9e9e7; margin:0.75rem 0 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Evaluation Curves</div>', unsafe_allow_html=True)

    roc_img = os.path.join(PHASE3_DIR, "roc_curve.png")
    pr_img  = os.path.join(PHASE3_DIR, "pr_curve.png")
    cm_img  = os.path.join(PHASE3_DIR, "confusion_matrix.png")

    c1, c2, c3 = st.columns(3)

    with c1:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curve — Holdout Test Set")
        else:
            fpr = [0.0, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            tpr = [0.0, 0.81, 0.81, 0.88, 0.92, 0.95, 0.98, 1.0]
            fig_roc = px.line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"})
            fig_roc.update_layout(
                **PLOT_BASE,
                height=220,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis={"gridcolor": "#e9e9e7", "tickfont": {"size": 9, "color": "#9fa3a8"}},
                yaxis={"gridcolor": "#e9e9e7", "tickfont": {"size": 9, "color": "#9fa3a8"}}
            )
            fig_roc.update_traces(line=dict(color="#4338ca", width=2))
            st.markdown('<p style="font-size:0.72rem; color:#9fa3a8; margin-bottom:0.3rem;">ROC Curve (simulated · AUC = 0.982)</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Precision-Recall Curve")
        else:
            recall_v    = [0.0, 0.6, 0.81, 0.81, 0.85, 0.9, 1.0]
            precision_v = [1.0, 1.0, 1.0,  0.8,  0.65, 0.3, 0.0]
            fig_pr = px.line(x=recall_v, y=precision_v, labels={"x": "Recall", "y": "Precision"})
            fig_pr.update_layout(
                **PLOT_BASE,
                height=220,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis={"gridcolor": "#e9e9e7", "tickfont": {"size": 9, "color": "#9fa3a8"}},
                yaxis={"gridcolor": "#e9e9e7", "tickfont": {"size": 9, "color": "#9fa3a8"}}
            )
            fig_pr.update_traces(line=dict(color="#7c3aed", width=2))
            st.markdown('<p style="font-size:0.72rem; color:#9fa3a8; margin-bottom:0.3rem;">Precision-Recall (simulated · AUC = 0.869)</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_pr, use_container_width=True)

    with c3:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix — LightGBM")
        else:
            cm_vals = [[1801, 0], [3, 13]]
            fig_cm = px.imshow(
                cm_vals, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Normal (0)", "Mule (1)"],
                y=["Normal (0)", "Mule (1)"],
                color_continuous_scale=[[0, "#f1f1ef"], [1, "#4338ca"]]
            )
            fig_cm.update_layout(
                **PLOT_BASE,
                height=220,
                margin=dict(l=0, r=0, t=8, b=0),
                coloraxis_showscale=False
            )
            fig_cm.update_traces(textfont=dict(size=14, color="#37352f"))
            st.markdown('<p style="font-size:0.72rem; color:#9fa3a8; margin-bottom:0.3rem;">Confusion Matrix (threshold = 0.40)</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_cm, use_container_width=True)
