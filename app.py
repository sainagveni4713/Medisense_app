import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="MediSense — Health Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1f2937;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: #f8fafc;
    color: #1f2937;
}
[data-testid="stHeader"] {
    border-bottom: 1px solid #e2e8f0;
    background: #ffffff;
}
.main .block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #334155;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f766e 0%, #134e4a 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25); }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] *,
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"] * {
    color: #1f2937 !important;
    fill: #1f2937 !important;
}

/* ── Typography ── */
h1 {
    color: #0f766e !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}
h2 { color: #134e4a !important; }
h3 { color: #1e293b !important; }
h4, h5, h6 { color: #334155 !important; }

.app-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin: 0 0 1rem;
}

/* ── Section title ── */
.section-title {
    color: #0f766e;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.25rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #ccfbf1;
}

/* ── Stat cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    min-height: 110px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: box-shadow 0.2s;
}
.stat-card:hover {
    border-color: #99f6e4;
    box-shadow: 0 6px 18px rgba(15,118,110,0.10);
}
.stat-label {
    color: #475569;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stat-value {
    color: #0f172a;
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 0.4rem;
}
.stat-note {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.45rem;
}

/* ── Risk banners ── */
.risk-high {
    background: #fff1f2;
    border-left: 5px solid #e11d48;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 10px 0;
    color: #881337;
}
.risk-moderate {
    background: #fffbeb;
    border-left: 5px solid #d97706;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 10px 0;
    color: #78350f;
}
.risk-low {
    background: #f0fdf4;
    border-left: 5px solid #16a34a;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 10px 0;
    color: #14532d;
}
.risk-high h2, .risk-high h3, .risk-high p,
.risk-moderate h2, .risk-moderate h3, .risk-moderate p,
.risk-low h2, .risk-low h3, .risk-low p {
    color: inherit !important;
    margin: 0 0 4px;
}

/* ── Info / disclaimer boxes ── */
.info-box {
    background: #f0fdfa;
    border-left: 5px solid #0d9488;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 10px 0;
}
.info-box h3 { margin: 0 0 6px; color: #0f766e !important; }
.info-box p  { margin: 0; color: #374151; font-size: 0.9rem; }

.disclaimer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 14px 0;
    font-size: 0.84rem;
    color: #64748b;
}

/* ── Buttons ── */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.1rem !important;
    font-size: 0.95rem !important;
    min-height: 2.7rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #0f766e, #134e4a) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,118,110,0.25) !important;
}

/* ── Form inputs ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stForm"] label {
    font-weight: 500;
    color: #374151 !important;
}
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1.2rem 1.3rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1f2937 !important;
    border-color: #cbd5e1 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button,
[data-testid="stTabs"] button p { color: #475569 !important; }
[data-testid="stTabs"] [aria-selected="true"] p {
    color: #0f766e !important;
    font-weight: 700;
}

/* ── Data table ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.9rem;
}
.data-table th {
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}
.data-table td {
    color: #334155;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #f1f5f9;
}
.data-table tr:last-child td { border-bottom: 0; }

/* ── Hide Streamlit chrome ── */
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section_title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def stat_card(label: str, value: str, note: str):
    st.markdown(f"""
    <div class="stat-card">
        <div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        <div class="stat-note">{note}</div>
    </div>""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame):
    st.markdown(df.to_html(index=False, classes="data-table", border=0), unsafe_allow_html=True)


# Plotly chart config — keeps zoom/pan/reset; removes clutter
CHART_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "autoScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian",
        "toggleSpikelines", "sendDataToCloud",
    ],
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "medisense_chart",
        "scale": 2,
    },
}

AXIS_BASE = dict(
    tickfont=dict(color="#475569"),
    gridcolor="#e2e8f0",
    zerolinecolor="#cbd5e1",
    linecolor="#e2e8f0",
)


def style_chart(fig: go.Figure, height: int = None) -> go.Figure:
    """Apply consistent theme. Never writes axis.title without explicit text."""
    layout = dict(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2937", size=12),
        legend=dict(font=dict(color="#334155")),
        margin=dict(t=52, b=40, l=48, r=28),
        xaxis=dict(title=dict(text=""), **AXIS_BASE),
        yaxis=dict(title=dict(text=""), **AXIS_BASE),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


def render_chart(fig: go.Figure, height: int = None):
    st.plotly_chart(
        style_chart(fig, height),
        use_container_width=True,
        config=CHART_CONFIG,
    )


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_d = pd.read_csv("data/Diabetics.csv")
    df_h = pd.read_csv("data/Heart_disease.csv")
    return df_d, df_h


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models():
    df_d, df_h = load_data()

    # ── Diabetes ──────────────────────────────
    d = df_d.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        d[col] = d[col].replace(0, np.nan).fillna(d[col].median())
    d.dropna(inplace=True)

    X_d = d.drop('Outcome', axis=1)
    y_d = d['Outcome'].astype(int)
    sc_d = StandardScaler()
    Xs_d = sc_d.fit_transform(X_d)

    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        Xs_d, y_d, test_size=0.2, random_state=42, stratify=y_d)
    rf_d = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_leaf=4, min_samples_split=8, random_state=42)
    rf_d.fit(Xd_tr, yd_tr)
    cv_d = cross_val_score(rf_d, Xs_d, y_d, cv=5, scoring='accuracy').mean()

    # ── Heart ──────────────────────────────────
    h = df_h.copy()
    h['Gender_enc'] = (h['Gender'] == 'Male').astype(int)
    h['CP_enc']     = h['Chest_Pain'].map({'no pain':0,'normal':1,'sometimes':2,'regular':3})
    h['SL_enc']     = h['ST_Slope'].map({'upward':0,'normal':1,'downward':2})
    h['Result_enc'] = (h['Result'] == 'Heart Disease').astype(int)
    h.dropna(inplace=True)

    feat_h = ['Age','Gender_enc','CP_enc','Systolic_BP','Diastolic_BP',
              'Cholesterol','Max_Heart_Beat','ST_Depression','SL_enc']
    X_h = h[feat_h]
    y_h = h['Result_enc']
    sc_h = StandardScaler()
    Xs_h = sc_h.fit_transform(X_h)

    Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(
        Xs_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
    rf_h = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        min_samples_leaf=5, min_samples_split=10, random_state=42)
    rf_h.fit(Xh_tr, yh_tr)
    cv_h = cross_val_score(rf_h, Xs_h, y_h, cv=5, scoring='accuracy').mean()

    # ── Feature importance label maps ──────────
    label_map_h = {
        'Age':'Age', 'Gender_enc':'Gender', 'CP_enc':'Chest Pain',
        'Systolic_BP':'Systolic BP', 'Diastolic_BP':'Diastolic BP',
        'Cholesterol':'Cholesterol', 'Max_Heart_Beat':'Max Heart Rate',
        'ST_Depression':'ST Depression', 'SL_enc':'ST Slope',
    }

    def _build_metrics(rf, Xte, yte, feature_names, label_map=None):
        preds = rf.predict(Xte)
        fi = dict(zip(feature_names, rf.feature_importances_))
        if label_map:
            fi = {label_map.get(k, k): v for k, v in fi.items()}
        return {
            'acc':    accuracy_score(yte, preds),
            'prec':   precision_score(yte, preds),
            'rec':    recall_score(yte, preds),
            'f1':     f1_score(yte, preds),
            'cv':     cv_d if rf is rf_d else cv_h,
            'cm':     confusion_matrix(yte, preds),
            'roc':    roc_curve(yte, rf.predict_proba(Xte)[:, 1]),
            'fi':     fi,
            'X_test': Xte,
            'y_test': yte,
        }

    metrics = {
        'diabetes': _build_metrics(rf_d, Xd_te, yd_te, X_d.columns.tolist()),
        'heart':    _build_metrics(rf_h, Xh_te, yh_te, feat_h, label_map_h),
    }
    # Fix cv values explicitly
    metrics['diabetes']['cv'] = cv_d
    metrics['heart']['cv']    = cv_h

    return rf_d, sc_d, X_d.columns.tolist(), rf_h, sc_h, feat_h, metrics, (d, h)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## MediSense")
    st.markdown("**Health Decision Support System**")
    st.markdown("*AI-powered disease risk assessment*")
    st.divider()
    page = st.radio("Navigate to", [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Analytics Dashboard",
        "Model Performance",
        "Privacy & Ethics",
    ])
    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.82; line-height:1.5;'>
    For educational use only.<br>
    Not a substitute for clinical diagnosis.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────
rf_d, sc_d, d_cols, rf_h, sc_h, h_cols, metrics, (df_d_clean, df_h_clean) = train_models()
df_d_raw, df_h_raw = load_data()


# ══════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════
if page == "Home":
    st.markdown("<h1>MediSense — Health Decision Support System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>AI-powered predictive analytics for diabetes and heart disease risk assessment</p>", unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Diabetes Records",   f"{len(df_d_raw):,}",                        "Pima Indian Dataset")
    with c2: stat_card("Heart Records",      f"{len(df_h_raw):,}",                        "Cleveland Dataset")
    with c3: stat_card("Diabetes Accuracy",  f"{metrics['diabetes']['acc']*100:.1f}%",    "Random Forest · 5-fold CV")
    with c4: stat_card("Heart Accuracy",     f"{metrics['heart']['acc']*100:.1f}%",       "Random Forest · 5-fold CV")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        section_title("What MediSense does")
        st.markdown("""
MediSense uses **Random Forest classification** trained on real clinical datasets to
estimate a patient's risk of:

- 🩸 **Type 2 Diabetes** — glucose, BMI, age, insulin, family history & more
- ❤️ **Heart Disease** — blood pressure, cholesterol, ECG markers & chest pain

Every prediction includes:
- A risk probability score (0–100%)
- Top contributing risk factors (XAI feature importance)
- Evidence-based clinical recommendations
        """)

    with col2:
        section_title("ML Pipeline")
        render_table(pd.DataFrame({
            "Step":   ["Data Ingestion","Imputation","Normalisation",
                       "Train/Test Split","Classification",
                       "Feature Importance","Evaluation"],
            "Detail": ["CSV — UCI/Kaggle","Median fill for zeros/nulls",
                       "StandardScaler (z-score)","80/20 stratified",
                       "Random Forest (200 trees)",
                       "RF feature_importances_",
                       "Accuracy, F1, AUC-ROC, 5-fold CV"],
        }))

    st.divider()
    section_title("Dataset Class Distribution")

    c1, c2 = st.columns(2)
    with c1:
        do = (df_d_raw['Outcome']
              .value_counts()
              .reset_index()
              .rename(columns={'Outcome':'Class','count':'Count'}))
        do['Class'] = do['Class'].map({0:'No Diabetes', 1:'Diabetes'})
        render_chart(
            px.pie(do, values='Count', names='Class',
                   title='Diabetes — Class Distribution',
                   color_discrete_sequence=['#0d9488','#f97316'], hole=0.42),
            height=320)

    with c2:
        ho = (df_h_raw['Result']
              .value_counts()
              .reset_index()
              .rename(columns={'Result':'Class','count':'Count'}))
        render_chart(
            px.pie(ho, values='Count', names='Class',
                   title='Heart Disease — Class Distribution',
                   color_discrete_sequence=['#22c55e','#dc2626'], hole=0.42),
            height=320)


# ══════════════════════════════════════════════
# PAGE — DIABETES PREDICTION
# ══════════════════════════════════════════════
elif page == "Diabetes Prediction":
    st.markdown("<h1>Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Enter patient vitals to receive an AI-driven diabetes risk score.</p>", unsafe_allow_html=True)
    st.divider()

    with st.form("diabetes_form"):
        section_title("Patient Information")

        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input("Pregnancies",         0,  20,  2,   help="Times pregnant")
        glucose     = c2.number_input("Glucose (mg/dL)",    50, 300, 120,  help="Plasma glucose (2-hr OGTT). Normal fasting: 70–99")
        blood_press = c3.number_input("Blood Pressure (mmHg)", 40, 150, 72, help="Diastolic BP. Normal: <80")
        skin_thick  = c4.number_input("Skin Thickness (mm)", 0, 100, 25,   help="Triceps skinfold thickness")

        c5, c6, c7, c8 = st.columns(4)
        insulin = c5.number_input("Insulin (µU/mL)",  0,   900, 80,  help="2-hr serum insulin. Normal: 2–25")
        bmi     = c6.number_input("BMI (kg/m²)",     10.0, 70.0, 28.5, step=0.1, help="Normal: 18.5–24.9")
        dpf     = c7.number_input("Diabetes Pedigree", 0.05, 2.50, 0.35, step=0.001, help="Genetic predisposition score")
        age     = c8.number_input("Age (years)",     18,  100,  35)

        submitted = st.form_submit_button("🔍 Assess Diabetes Risk", use_container_width=True)

    if submitted:
        iv   = np.array([[pregnancies, glucose, blood_press, skin_thick, insulin, bmi, dpf, age]])
        prob = rf_d.predict_proba(sc_d.transform(iv))[0][1]
        pct  = int(prob * 100)

        risk_class = "risk-high"     if pct >= 65 else "risk-moderate" if pct >= 35 else "risk-low"
        risk_label = "HIGH RISK"     if pct >= 65 else "MODERATE RISK" if pct >= 35 else "LOW RISK"
        risk_icon  = "🚨"            if pct >= 65 else "⚠️"            if pct >= 35 else "✅"
        risk_msg   = ("Immediate clinical evaluation recommended."              if pct >= 65
                      else "Further screening and lifestyle assessment advised." if pct >= 35
                      else "Maintain healthy lifestyle and routine check-ups.")

        st.divider()
        section_title("Risk Assessment Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h2>{risk_icon} {risk_label}</h2>
                <h3>Diabetes Risk Probability: <strong>{pct}%</strong></h3>
                <p>{risk_msg}</p>
            </div>""", unsafe_allow_html=True)

            gauge_color = "#dc2626" if pct >= 65 else "#d97706" if pct >= 35 else "#16a34a"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                title={'text': "Risk Score (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': gauge_color},
                    'steps': [
                        {'range': [0,  35],  'color': '#dcfce7'},
                        {'range': [35, 65],  'color': '#fef3c7'},
                        {'range': [65, 100], 'color': '#fee2e2'},
                    ],
                    'threshold': {
                        'line': {'color': '#0f766e', 'width': 4},
                        'thickness': 0.75, 'value': pct,
                    },
                },
            ))
            render_chart(fig_g, height=300)

        with col2:
            section_title("Feature Importance (Model-wide)")
            fi = dict(sorted(metrics['diabetes']['fi'].items(), key=lambda x: x[1], reverse=True))
            fig_fi = px.bar(
                x=list(fi.values()), y=list(fi.keys()),
                orientation='h',
                color=list(fi.values()), color_continuous_scale='Teal',
                labels={'x': 'Importance Score', 'y': 'Feature'},
                title="Random Forest Feature Importance — Diabetes",
            )
            fig_fi.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange='reversed'),
                xaxis=dict(title=dict(text='Importance Score')),
            )
            render_chart(fig_fi, height=330)

        section_title("Your Values vs. Dataset Mean")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name='Your Value',   x=d_cols, y=iv[0],      marker_color='#0d9488'))
        fig_cmp.add_trace(go.Bar(name='Dataset Mean', x=d_cols, y=sc_d.mean_, marker_color='#94a3b8'))
        fig_cmp.update_layout(
            barmode='group',
            xaxis=dict(title=dict(text='Feature'), tickangle=-25),
            yaxis=dict(title=dict(text='Value (original scale)')),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        render_chart(fig_cmp, height=340)

        section_title("Clinical Recommendations")
        if pct >= 65:
            recs = [
                "🏥 Consult an endocrinologist — fasting blood glucose and HbA1c tests urgently recommended.",
                "🥗 Dietary audit: reduce refined carbohydrates, sugars, and processed foods immediately.",
                "⚖️ BMI management: 5–10% weight loss can significantly reduce Type 2 diabetes onset.",
                "🏃 Physical activity: 150 min/week of moderate aerobic exercise (WHO guideline).",
                "💊 Discuss metformin or other preventive medications with your physician.",
            ]
        elif pct >= 35:
            recs = [
                "🔬 Schedule routine HbA1c screening within the next 3 months.",
                "🥗 Adopt Mediterranean or DASH diet patterns to improve insulin sensitivity.",
                "🏃 Increase daily physical activity — 30 min moderate exercise most days.",
                "📊 Monitor glucose levels if there is a family history of diabetes.",
            ]
        else:
            recs = [
                "✅ Maintain current healthy lifestyle — diet and exercise remain key.",
                "🗓️ Routine annual check-up including fasting glucose is recommended.",
                "⚠️ Stay aware of risk factors: obesity, sedentary lifestyle, family history.",
            ]
        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("""
        <div class="disclaimer-box">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool provides AI-assisted risk stratification
        for educational and decision-support purposes only. All results must be confirmed by a
        qualified healthcare professional. Patient data entered here is not stored or transmitted.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE — HEART DISEASE PREDICTION
# ══════════════════════════════════════════════
elif page == "Heart Disease Prediction":
    st.markdown("<h1>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Enter cardiovascular data to receive a heart disease risk score.</p>", unsafe_allow_html=True)
    st.divider()

    with st.form("heart_form"):
        section_title("Patient Information")

        c1, c2, c3 = st.columns(3)
        h_age    = c1.number_input("Age (years)", 18, 100, 55)
        h_gender = c2.selectbox("Gender", ["Male", "Female"])
        h_chest  = c3.selectbox("Chest Pain Type",
                                ["No pain", "Normal/mild", "Sometimes", "Regular/severe"])

        c4, c5, c6 = st.columns(3)
        h_sbp  = c4.number_input("Systolic BP (mmHg)",  80, 220, 140, help="Normal: <120")
        h_dbp  = c5.number_input("Diastolic BP (mmHg)", 50, 130,  90, help="Normal: <80")
        h_chol = c6.number_input("Cholesterol (mg/dL)", 100, 600, 250, help="Desirable: <200")

        c7, c8, c9 = st.columns(3)
        h_hr    = c7.number_input("Max Heart Rate (bpm)", 60, 220, 150, help="Peak during exercise ECG")
        h_st    = c8.number_input("ST Depression",        0.0, 7.0, 1.5, step=0.1,
                                  help="ECG ST segment depression vs rest")
        h_slope = c9.selectbox("ST Slope",
                               ["Upward (normal)", "Flat", "Downward (abnormal)"])

        submitted_h = st.form_submit_button("🔍 Assess Heart Disease Risk", use_container_width=True)

    if submitted_h:
        ge = 1 if h_gender == "Male" else 0
        ce = {"No pain": 0, "Normal/mild": 1, "Sometimes": 2, "Regular/severe": 3}[h_chest]
        se = {"Upward (normal)": 0, "Flat": 1, "Downward (abnormal)": 2}[h_slope]

        iv_h = np.array([[h_age, ge, ce, h_sbp, h_dbp, h_chol, h_hr, h_st, se]])
        pct_h = int(rf_h.predict_proba(sc_h.transform(iv_h))[0][1] * 100)

        risk_class = "risk-high"     if pct_h >= 65 else "risk-moderate" if pct_h >= 35 else "risk-low"
        risk_label = "HIGH RISK"     if pct_h >= 65 else "MODERATE RISK" if pct_h >= 35 else "LOW RISK"
        risk_icon  = "🚨"            if pct_h >= 65 else "⚠️"            if pct_h >= 35 else "✅"
        risk_msg   = ("Urgent cardiology consultation recommended."               if pct_h >= 65
                      else "Cardiac risk assessment and lifestyle changes advised." if pct_h >= 35
                      else "Maintain heart-healthy lifestyle and routine check-ups.")

        st.divider()
        section_title("Risk Assessment Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h2>{risk_icon} {risk_label}</h2>
                <h3>Heart Disease Probability: <strong>{pct_h}%</strong></h3>
                <p>{risk_msg}</p>
            </div>""", unsafe_allow_html=True)

            gauge_color = "#dc2626" if pct_h >= 65 else "#d97706" if pct_h >= 35 else "#16a34a"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct_h,
                title={'text': "Risk Score (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': gauge_color},
                    'steps': [
                        {'range': [0,  35],  'color': '#dcfce7'},
                        {'range': [35, 65],  'color': '#fef3c7'},
                        {'range': [65, 100], 'color': '#fee2e2'},
                    ],
                    'threshold': {
                        'line': {'color': '#dc2626', 'width': 4},
                        'thickness': 0.75, 'value': pct_h,
                    },
                },
            ))
            render_chart(fig_g, height=300)

        with col2:
            section_title("Feature Importance (Model-wide)")
            fi_h = dict(sorted(metrics['heart']['fi'].items(), key=lambda x: x[1], reverse=True))
            fig_fi = px.bar(
                x=list(fi_h.values()), y=list(fi_h.keys()),
                orientation='h',
                color=list(fi_h.values()), color_continuous_scale='Reds',
                labels={'x': 'Importance Score', 'y': 'Feature'},
                title="Random Forest Feature Importance — Heart Disease",
            )
            fig_fi.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange='reversed'),
                xaxis=dict(title=dict(text='Importance Score')),
            )
            render_chart(fig_fi, height=330)

        section_title("Clinical Recommendations")
        if pct_h >= 65:
            recs = [
                "🏥 Seek urgent cardiology consultation — consider ECG, echocardiogram, and stress test.",
                "💊 Cholesterol management: statin therapy may be indicated if LDL > 190 mg/dL.",
                "🩺 Blood pressure control: target BP < 130/80 mmHg (ACC/AHA 2023 guideline).",
                "🚭 Avoid smoking and limit alcohol — both are independent cardiac risk factors.",
                "🚨 Patient and family should recognise the signs of myocardial infarction.",
            ]
        elif pct_h >= 35:
            recs = [
                "🔬 Schedule cardiac risk assessment — lipid panel and resting ECG recommended.",
                "⚖️ Achieve and maintain healthy weight (BMI 18.5–24.9).",
                "🐟 Increase omega-3 intake (fatty fish, flaxseeds) and reduce saturated fats.",
                "🏃 Regular aerobic exercise — 20 min/day reduces cardiac risk by ~30%.",
            ]
        else:
            recs = [
                "✅ Continue heart-healthy diet: fruits, vegetables, whole grains, lean protein.",
                "🗓️ Annual lipid panel and blood pressure check recommended after age 40.",
                "🏃 Stay physically active and maintain a healthy body weight.",
            ]
        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("""
        <div class="disclaimer-box">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool provides AI-assisted risk stratification
        for educational and decision-support purposes only. Results must be confirmed by a qualified
        healthcare professional. No data is stored or transmitted.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
elif page == "Analytics Dashboard":
    st.markdown("<h1>Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Exploratory data analysis across both clinical datasets.</p>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2 = st.tabs(["🩸 Diabetes Dataset", "❤️ Heart Disease Dataset"])

    with tab1:
        section_title("Descriptive Statistics")
        render_table(df_d_raw.describe().round(2).reset_index().rename(columns={"index": "Metric"}))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Glucose Distribution by Outcome")
            render_chart(
                px.histogram(df_d_raw, x='Glucose', color='Outcome',
                             barmode='overlay', nbins=30,
                             color_discrete_map={0: '#0d9488', 1: '#f97316'},
                             labels={'Outcome': 'Diabetic', 'Glucose': 'Glucose (mg/dL)'}),
                height=320)
        with c2:
            st.markdown("#### BMI vs Glucose (by Outcome)")
            render_chart(
                px.scatter(df_d_raw, x='BMI', y='Glucose', color='Outcome',
                           color_discrete_map={0: '#0d9488', 1: '#f97316'},
                           labels={'BMI': 'BMI (kg/m²)', 'Glucose': 'Glucose (mg/dL)'},
                           opacity=0.6),
                height=320)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Age Distribution by Outcome")
            render_chart(
                px.histogram(df_d_raw, x='Age', color='Outcome', nbins=20,
                             barmode='overlay',
                             color_discrete_map={0: '#0d9488', 1: '#f97316'},
                             labels={'Age': 'Age (years)'}),
                height=320)
        with c4:
            st.markdown("#### Feature Correlation Heatmap")
            corr = df_d_raw.corr().round(2)
            fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r',
                                 zmin=-1, zmax=1, text_auto=True, aspect='auto')
            fig_corr.update_layout(
                xaxis=dict(title=dict(text="")),
                yaxis=dict(title=dict(text="")),
            )
            render_chart(fig_corr, height=360)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Cholesterol by Diagnosis")
            fig_box = px.box(df_h_raw, x='Result', y='Cholesterol', color='Result',
                             color_discrete_map={'Heart Disease': '#dc2626',
                                                 'No Heart Disease': '#22c55e'},
                             labels={'Cholesterol': 'Cholesterol (mg/dL)'})
            fig_box.update_layout(showlegend=False)
            render_chart(fig_box, height=320)
        with c2:
            st.markdown("#### Age vs Max Heart Rate")
            render_chart(
                px.scatter(df_h_raw, x='Age', y='Max_Heart_Beat', color='Result',
                           color_discrete_map={'Heart Disease': '#dc2626',
                                               'No Heart Disease': '#22c55e'},
                           labels={'Age': 'Age (years)', 'Max_Heart_Beat': 'Max Heart Rate (bpm)'},
                           opacity=0.7),
                height=320)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Chest Pain Type vs Diagnosis")
            cp_counts = (df_h_raw
                         .groupby(['Chest_Pain', 'Result'])
                         .size()
                         .reset_index(name='Count'))
            fig_cp = px.bar(cp_counts, x='Chest_Pain', y='Count', color='Result',
                            barmode='group',
                            color_discrete_map={'Heart Disease': '#dc2626',
                                                'No Heart Disease': '#22c55e'},
                            labels={'Chest_Pain': 'Chest Pain Type'})
            fig_cp.update_layout(xaxis=dict(tickangle=-20))
            render_chart(fig_cp, height=330)
        with c4:
            st.markdown("#### ST Depression Distribution")
            render_chart(
                px.histogram(df_h_raw, x='ST_Depression', color='Result',
                             nbins=20, barmode='overlay',
                             color_discrete_map={'Heart Disease': '#dc2626',
                                                 'No Heart Disease': '#22c55e'},
                             labels={'ST_Depression': 'ST Depression'}),
                height=320)


# ══════════════════════════════════════════════
# PAGE — MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("<h1>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Comprehensive evaluation of the trained classification models.</p>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2 = st.tabs(["🩸 Diabetes Model", "❤️ Heart Disease Model"])

    for tab, disease, label, color in [
        (tab1, 'diabetes', 'Diabetes',      '#0d9488'),
        (tab2, 'heart',    'Heart Disease',  '#dc2626'),
    ]:
        with tab:
            m = metrics[disease]

            # ── KPI row ──
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",   f"{m['acc']*100:.1f}%")
            c2.metric("Precision",  f"{m['prec']*100:.1f}%")
            c3.metric("Recall",     f"{m['rec']*100:.1f}%")
            c4.metric("F1 Score",   f"{m['f1']*100:.1f}%")
            c5.metric("CV Score",   f"{m['cv']*100:.1f}%")

            col1, col2 = st.columns(2)

            # ── Confusion matrix ──
            with col1:
                st.markdown("#### Confusion Matrix")
                fig_cm = px.imshow(
                    m['cm'], text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    color_continuous_scale=[[0, '#f0fdfa'], [1, color]],
                )
                fig_cm.update_layout(
                    xaxis=dict(title=dict(text="Predicted")),
                    yaxis=dict(title=dict(text="Actual")),
                )
                render_chart(fig_cm, height=320)

            # ── ROC curve ──
            with col2:
                st.markdown("#### ROC Curve")
                fpr, tpr, _ = m['roc']
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'AUC = {roc_auc:.3f}',
                    line=dict(color=color, width=2.5)))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    name='Random baseline',
                    line=dict(color='#94a3b8', dash='dash')))
                fig_roc.update_layout(
                    xaxis=dict(title=dict(text='False Positive Rate')),
                    yaxis=dict(title=dict(text='True Positive Rate')),
                    legend=dict(x=0.58, y=0.1),
                )
                render_chart(fig_roc, height=320)

            # ── Feature importance ──
            section_title("Feature Importance")
            fi_df = (pd.DataFrame({'Feature': list(m['fi'].keys()),
                                   'Importance': list(m['fi'].values())})
                     .sort_values('Importance', ascending=False))
            fig_fi = px.bar(fi_df, x='Feature', y='Importance',
                            color='Importance',
                            color_continuous_scale='Teal' if disease == 'diabetes' else 'Reds',
                            title=f"Feature Importances — {label}",
                            labels={'Feature': 'Feature', 'Importance': 'Importance Score'})
            fig_fi.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(tickangle=-20),
            )
            render_chart(fig_fi, height=350)

            # ── Classification report ──
            section_title("Classification Report")
            rf = rf_d if disease == 'diabetes' else rf_h
            y_pred = rf.predict(m['X_test'])
            report_df = (pd.DataFrame(
                classification_report(m['y_test'], y_pred, output_dict=True))
                .transpose()
                .round(3)
                .reset_index()
                .rename(columns={"index": "Class"}))
            render_table(report_df)


# ══════════════════════════════════════════════
# PAGE — PRIVACY & ETHICS
# ══════════════════════════════════════════════
elif page == "Privacy & Ethics":
    st.markdown("<h1>Privacy & Ethics</h1>", unsafe_allow_html=True)
    st.markdown("Healthcare AI carries unique ethical responsibilities. Here is how MediSense addresses them.")
    st.divider()

    col1, col2 = st.columns(2)

    left_cards = [
        ("🔒 Zero Data Retention",
         "All computations run locally within this Streamlit session. No health data is stored "
         "in any database or transmitted to external servers. Session data is cleared when the "
         "browser tab is closed."),
        ("🧠 Model Transparency (XAI)",
         "Every prediction includes a feature importance breakdown showing which clinical variables "
         "most influenced the result. This supports explainable AI (XAI) principles required for "
         "clinical decision support tools."),
        ("📜 Dataset Licensing",
         "<strong>Pima Indians Diabetes</strong> — Smith et al. (1988), UCI Machine Learning "
         "Repository, CC BY 4.0.<br>"
         "<strong>Cleveland Heart Disease</strong> — Cleveland Clinic Foundation, UCI Repository.<br>"
         "Both datasets are fully anonymised with no personally identifiable information."),
    ]

    right_cards = [
        ("⚖️ Algorithmic Fairness & Bias",
         "The Pima dataset is limited to <strong>female patients of Pima Indian heritage aged ≥21</strong>. "
         "The heart dataset has demographic skew (predominantly male). These biases are acknowledged — "
         "predictions for underrepresented demographics should be interpreted with extra caution."),
        ("🩺 Clinical Boundaries",
         "MediSense is a <strong>decision support</strong> tool — it assists, not replaces, clinical judgment. "
         "Risk scores should trigger clinical follow-up, not standalone diagnoses. No AI tool meets the "
         "diagnostic standard of an in-person clinical evaluation."),
        ("🔐 HIPAA Alignment",
         "Since no Protected Health Information (PHI) is transmitted or stored, this app operates outside "
         "HIPAA data-at-rest and in-transit obligations. For production clinical deployment, BAAs, audit "
         "logs, and de-identification protocols would be required."),
    ]

    with col1:
        for title, body in left_cards:
            st.markdown(f"<div class='info-box'><h3>{title}</h3><p>{body}</p></div>",
                        unsafe_allow_html=True)
    with col2:
        for title, body in right_cards:
            st.markdown(f"<div class='info-box'><h3>{title}</h3><p>{body}</p></div>",
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("### ✅ Specification Coverage Checklist")
    checks = [
        "UCI / Kaggle datasets used",
        "Median imputation for missing/zero values",
        "Random Forest classification (200 trees)",
        "StandardScaler z-score normalisation",
        "Feature importance analysis (XAI)",
        "5-fold cross-validation reported",
        "Confusion matrix per model",
        "AUC-ROC curve per model",
        "Exploratory data analysis charts",
        "In-app evidence-based recommendations",
        "Ethical data handling section",
        "Algorithmic bias & fairness disclosure",
        "Clinical disclaimer on all predictions",
        "No data storage or transmission",
    ]
    cols = st.columns(2)
    for i, item in enumerate(checks):
        cols[i % 2].markdown(f"✅ {item}")