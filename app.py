import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f5ede8 !important;
    color: #2c1810;
}
.stApp { background-color: #f5ede8 !important; }

#MainMenu, header, footer, [data-testid="collapsedControl"] {
    visibility: hidden !important;
    display: none !important;
}

/* ── Top badge ── */
.top-badge { text-align:center; margin-bottom:0.8rem; }
.top-badge span {
    background:transparent; border:1.5px solid #c0826a; color:#c0826a;
    font-size:0.7rem; font-weight:600; letter-spacing:0.15em;
    padding:5px 16px; border-radius:20px; text-transform:uppercase;
}

/* ── Hero ── */
.hero-title { text-align:center; margin-bottom:0.3rem; }
.hero-title h1 {
    font-family:'Playfair Display',serif;
    font-size:3.2rem; font-weight:800; line-height:1.1; margin:0;
}
.line1 { color:#2c1810; }
.line2 { color:#c0392b; }
.hero-subtitle {
    text-align:center; color:#7a6058; font-size:0.92rem;
    margin-bottom:1.8rem; line-height:1.5;
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid #e0d0c8; margin:1.2rem 0; }

/* ── Section heading ── */
.sec-heading {
    display:flex; align-items:center; gap:8px; margin-bottom:0.8rem;
}
.sec-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:24px; height:24px; background:#c0392b; color:white;
    border-radius:6px; font-size:0.78rem; font-weight:700; flex-shrink:0;
}
.sec-title { font-size:0.95rem; font-weight:600; color:#2c1810; }

/* ── Number inputs ── */
.stNumberInput label, .stSelectbox label {
    font-size:0.72rem !important; font-weight:600 !important;
    letter-spacing:0.07em !important; text-transform:uppercase !important;
    color:#7a6058 !important;
}
.stNumberInput input {
    background:white !important; border:1px solid #e0d0c8 !important;
    border-radius:8px !important; color:#2c1810 !important;
    font-size:1rem !important; font-weight:500 !important;
}
.stSelectbox > div > div {
    background:white !important; border:1px solid #e0d0c8 !important;
    border-radius:8px !important; color:#2c1810 !important;
}

/* ── Stat boxes ── */
.stat-row { display:flex; gap:8px; margin:0.8rem 0; }
.stat-box {
    flex:1; background:white; border:1px solid #e8d8d0;
    border-radius:10px; padding:0.6rem 0.4rem; text-align:center;
}
.stat-box .slbl {
    font-size:0.62rem; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; color:#c0826a; margin-bottom:3px;
}
.stat-box .sval { font-size:1.25rem; font-weight:700; color:#c0392b; }

/* ── Predict button ── */
.stButton > button {
    background:#c0392b !important; color:white !important;
    border:none !important; border-radius:10px !important;
    font-size:0.85rem !important; font-weight:700 !important;
    letter-spacing:0.08em !important; text-transform:uppercase !important;
    padding:0.75rem 2rem !important; width:100% !important;
}
.stButton > button:hover { background:#a93226 !important; }

/* ── Result banners ── */
.result-high {
    background:#fdf2f0; border:1.5px solid #e74c3c; border-left:5px solid #e74c3c;
    border-radius:10px; padding:1.1rem 1.4rem;
    display:flex; align-items:flex-start; gap:12px; margin:1rem 0;
}
.result-low {
    background:#f0fdf4; border:1.5px solid #27ae60; border-left:5px solid #27ae60;
    border-radius:10px; padding:1.1rem 1.4rem;
    display:flex; align-items:flex-start; gap:12px; margin:1rem 0;
}
.rtitle { font-size:0.95rem; font-weight:700; margin-bottom:4px; }
.rbody  { font-size:0.83rem; color:#555; line-height:1.5; }

/* ── Info section wrapper ── */
.info-card {
    background:#fff8f5; border:1px solid #e8d8d0; border-radius:14px;
    padding:1.3rem 1.6rem; margin:0.8rem 0;
}
.info-card-title {
    font-size:0.75rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:#7a6058; margin-bottom:0.9rem;
}

/* ── Health tips ── */
.tips-header { font-size:1.1rem; font-weight:700; color:#2c1810; margin:1.5rem 0 0.8rem; }
.tips-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1.2rem; }
.tip-card {
    background:white; border:1px solid #e8d8d0; border-radius:12px; padding:1rem;
}
.tip-card-red {
    background:#fff8f5; border:1px solid #f5c0b0; border-radius:12px; padding:1rem;
}
.tip-icon  { font-size:1.4rem; margin-bottom:0.4rem; }
.tip-title { font-size:0.85rem; font-weight:700; color:#2c1810; margin-bottom:0.3rem; }
.tip-body  { font-size:0.74rem; color:#7a6058; line-height:1.5; }

/* ── Disclaimer ── */
.disclaimer {
    background:#fffaf7; border:1px solid #e8d8d0; border-radius:10px;
    padding:0.85rem 1.2rem; font-size:0.77rem; color:#7a6058;
    margin:0.8rem 0; line-height:1.6;
}

/* ── Footer ── */
.footer {
    text-align:center; color:#b0988e; font-size:0.74rem;
    padding:1.2rem 0 0.5rem; border-top:1px solid #e0d0c8; margin-top:1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — build one risk-factor card as HTML string (NO f-string CSS braces)
# ─────────────────────────────────────────────────────────────────────────────
def rf_card(icon, label, value, status, note):
    bg  = {"high":"#fff0ee",  "medium":"#fffbec",  "low":"#f0fdf4"}[status]
    bdr = {"high":"#f5c0b0",  "medium":"#f5e0a0",  "low":"#a8e6bc"}[status]
    clr = {"high":"#c0392b",  "medium":"#b7770d",  "low":"#1e8449"}[status]
    return (
        '<div style="background:' + bg + ';border:1px solid ' + bdr + ';border-radius:10px;'
        'padding:0.75rem 0.9rem;display:flex;justify-content:space-between;'
        'align-items:center;gap:8px;">'
          '<div style="display:flex;align-items:center;gap:8px;min-width:0;">'
            '<span style="font-size:0.9rem;">' + icon + '</span>'
            '<div>'
              '<div style="font-size:0.78rem;font-weight:600;color:#2c1810;">' + label + '</div>'
              '<div style="font-size:0.69rem;color:#7a6058;margin-top:1px;">' + note + '</div>'
            '</div>'
          '</div>'
          '<div style="font-size:0.85rem;font-weight:700;color:' + clr + ';'
          'white-space:nowrap;flex-shrink:0;">' + value + '</div>'
        '</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_extended_10000.csv")
    df["AlcoholConsumption"] = df["AlcoholConsumption"].fillna("None")
    return df

@st.cache_data
def run_pipeline(df):
    df = df.copy()
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    for col in ["Sex","ExerciseAngina"]:
        X[col] = LabelEncoder().fit_transform(X[col])
    for col, mp in {
        "SmokingStatus":         {"Never":0,"Former":1,"Current":2},
        "PhysicalActivityLevel": {"Low":0,"Moderate":1,"High":2},
        "AlcoholConsumption":    {"None":0,"Moderate":1,"Heavy":2},
        "StressLevel":           {"Low":0,"Moderate":1,"High":2},
    }.items():
        X[col] = X[col].map(mp)
    X = pd.get_dummies(X, columns=["ChestPainType","RestingECG","ST_Slope"], drop_first=True)
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sel.fit(X, y)
    imp = pd.Series(rf_sel.feature_importances_, index=X.columns)
    sel = imp[imp >= 0.01].index.tolist()
    X = X[sel]
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_tr_s = pd.DataFrame(scaler.fit_transform(x_tr), columns=x_tr.columns)
    x_te_s  = pd.DataFrame(scaler.transform(x_te),    columns=x_te.columns)
    x_sm, y_sm = SMOTE(random_state=42).fit_resample(x_tr_s, y_tr)
    return x_tr_s, x_te_s, y_tr, y_te, x_sm, y_sm, scaler, sel

@st.cache_data
def train_best_model(x_sm, y_sm, _x_te, y_te, _sel):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, use_label_encoder=False,
                                              eval_metric="logloss", random_state=42),
    }
    best_name, best_model, best_f1 = "", None, 0
    for name, m in models.items():
        m.fit(x_sm, y_sm)
        f1 = f1_score(y_te, m.predict(_x_te))
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, m
    return best_name, best_model

def prep_input(ui, scaler, sel):
    d = {
        "Age":ui["Age"],"RestingBP":ui["RestingBP"],"Cholesterol":ui["Cholesterol"],
        "FastingBS":ui["FastingBS"],"MaxHR":ui["MaxHR"],"Oldpeak":ui["Oldpeak"],
        "BMI":ui["BMI"],"Diabetes":ui["Diabetes"],"BloodSugar_mg_dL":ui["BloodSugar_mg_dL"],
        "HDL_mg_dL":ui["HDL_mg_dL"],"LDL_mg_dL":ui["LDL_mg_dL"],
        "Triglycerides_mg_dL":ui["Triglycerides_mg_dL"],"FamilyHistoryHD":ui["FamilyHistoryHD"],
        "SleepHoursPerNight":ui["SleepHoursPerNight"],"CreatinineLevel":ui["CreatinineLevel"],
        "CRP_mg_L":ui["CRP_mg_L"],
        "Sex":            1 if ui["Sex"]=="M" else 0,
        "ExerciseAngina": 1 if ui["ExerciseAngina"]=="Y" else 0,
        "SmokingStatus":         {"Never":0,"Former":1,"Current":2}[ui["SmokingStatus"]],
        "PhysicalActivityLevel": {"Low":0,"Moderate":1,"High":2}[ui["PhysicalActivityLevel"]],
        "AlcoholConsumption":    {"None":0,"Moderate":1,"Heavy":2}[ui["AlcoholConsumption"]],
        "StressLevel":           {"Low":0,"Moderate":1,"High":2}[ui["StressLevel"]],
        "ChestPainType_ATA": int(ui["ChestPainType"]=="ATA"),
        "ChestPainType_NAP": int(ui["ChestPainType"]=="NAP"),
        "ChestPainType_TA":  int(ui["ChestPainType"]=="TA"),
        "RestingECG_LVH":    int(ui["RestingECG"]=="LVH"),
        "RestingECG_ST":     int(ui["RestingECG"]=="ST"),
        "ST_Slope_Flat":     int(ui["ST_Slope"]=="Flat"),
        "ST_Slope_Up":       int(ui["ST_Slope"]=="Up"),
    }
    row = pd.DataFrame([d])
    for c in sel:
        if c not in row.columns: row[c] = 0
    row = row[sel]
    return pd.DataFrame(scaler.transform(row), columns=row.columns)

# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────
df_raw = load_data()
with st.spinner("Initialising models… (first run only)"):
    x_tr_s, x_te_s, y_tr, y_te, x_sm, y_sm, scaler, sel_feat = run_pipeline(df_raw)
    best_name, best_model = train_best_model(x_sm, y_sm, x_te_s, y_te, tuple(sel_feat))

# ─────────────────────────────────────────────────────────────────────────────
# HEADER  — badge + title moved closer together
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="top-badge"><span>🫀 Clinical AI Decision Support</span></div>'
    '<div class="hero-title"><h1>'
    '<span class="line1">Heart Disease</span><br>'
    '<span class="line2">Prediction System</span>'
    '</h1></div>'
    '<div class="hero-subtitle">'
    'Enter patient clinical parameters to receive an instant AI-powered<br>cardiac risk assessment.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION HEADINGS  — rendered ABOVE the columns so they align perfectly
# ─────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns(2, gap="large")
with h1:
    st.markdown(
        '<div class="sec-heading">'
        '<span class="sec-num">1</span>'
        '<span class="sec-title">Patient Vitals</span>'
        '</div>',
        unsafe_allow_html=True
    )
with h2:
    st.markdown(
        '<div class="sec-heading">'
        '<span class="sec-num">2</span>'
        '<span class="sec-title">Clinical Findings</span>'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLUMNS  — equal number of rows so both columns are the same height
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    # Row 1
    r1a, r1b = st.columns(2)
    age  = r1a.number_input("Age (Years)",          min_value=18,  max_value=100, value=30,   step=1)
    rbp  = r1b.number_input("Resting BP (mmHg)",    min_value=80,  max_value=220, value=120,  step=1)
    # Row 2
    r2a, r2b = st.columns(2)
    chol  = r2a.number_input("Cholesterol (mg/dL)", min_value=85,  max_value=400, value=150,  step=1)
    maxhr = r2b.number_input("Max Heart Rate (BPM)",min_value=60,  max_value=202, value=150,  step=1)
    # Row 3
    r3a, r3b = st.columns(2)
    oldpk = r3a.number_input("Oldpeak (ST Depr.)",  min_value=0.0, max_value=6.2, value=0.5,  step=0.1)
    fbs   = r3b.selectbox("Fasting BS > 120?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    # Row 4
    r4a, r4b = st.columns(2)
    bmi  = r4a.number_input("BMI (kg/m²)",          min_value=10.0,max_value=60.0,value=24.0, step=0.1)
    sex  = r4b.selectbox("Sex", ["M","F"], format_func=lambda x: "Male" if x=="M" else "Female")
    # Row 5
    r5a, r5b = st.columns(2)
    hdl  = r5a.number_input("HDL Cholesterol",      min_value=10,  max_value=100, value=55,   step=1)
    ldl  = r5b.number_input("LDL Cholesterol",      min_value=30,  max_value=300, value=100,  step=1)
    # Row 6
    r6a, r6b = st.columns(2)
    trig = r6a.number_input("Triglycerides",        min_value=50,  max_value=500, value=140,  step=1)
    bs   = r6b.number_input("Blood Sugar (mg/dL)",  min_value=60,  max_value=300, value=90,   step=1)
    # Row 7
    r7a, r7b = st.columns(2)
    creat= r7a.number_input("Creatinine (mg/dL)",   min_value=0.4, max_value=5.0, value=1.0,  step=0.01)
    crp  = r7b.number_input("CRP (mg/L)",           min_value=0.1, max_value=15.0,value=1.0,  step=0.1)

with col2:
    # Row 1
    cp = st.selectbox("Chest Pain Type", ["ASY","ATA","NAP","TA"], format_func=lambda x: {
        "ASY":"ASY — Asymptomatic","ATA":"ATA — Atypical Angina",
        "NAP":"NAP — Non-Anginal Pain","TA":"TA — Typical Angina"}[x])
    # Row 2
    ecg = st.selectbox("Resting ECG Result", ["Normal","ST","LVH"], format_func=lambda x: {
        "Normal":"Normal","ST":"ST — ST-T Wave Abnormality",
        "LVH":"LVH — Left Ventricular Hypertrophy"}[x])
    # Row 3
    slope = st.selectbox("ST Slope (Exercise)", ["Up","Flat","Down"], format_func=lambda x: {
        "Up":"Up — Upsloping","Flat":"Flat — Flat","Down":"Down — Downsloping"}[x])
    # Row 4
    ea = st.selectbox("Exercise Angina", ["N","Y"],
                      format_func=lambda x: "No" if x=="N" else "Yes")
    # Row 5
    smoking  = st.selectbox("Smoking Status",      ["Never","Former","Current"])
    # Row 6
    activity = st.selectbox("Physical Activity",   ["Low","Moderate","High"])
    # Row 7
    r8a, r8b = st.columns(2)
    alcohol  = r8a.selectbox("Alcohol",            ["None","Moderate","Heavy"])
    stress   = r8b.selectbox("Stress Level",       ["Low","Moderate","High"])
    # Row 8
    r9a, r9b = st.columns(2)
    sleep    = r9a.number_input("Sleep (hrs/night)",min_value=3.0,max_value=12.0,value=7.0,step=0.5)
    diabetes = r9b.selectbox("Diabetes",           [0,1], format_func=lambda x:"Yes" if x else "No")
    # Row 9
    family_hd = st.selectbox("Family History of Heart Disease", [0,1],
                              format_func=lambda x:"Yes" if x else "No")

# ─────────────────────────────────────────────────────────────────────────────
# STAT BOXES + PREDICT BUTTON  — full width, below both columns
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="stat-row">'
    '<div class="stat-box"><div class="slbl">Age</div><div class="sval">' + str(age) + '</div></div>'
    '<div class="stat-box"><div class="slbl">BP</div><div class="sval">' + str(rbp) + '</div></div>'
    '<div class="stat-box"><div class="slbl">Cholesterol</div><div class="sval">' + str(chol) + '</div></div>'
    '<div class="stat-box"><div class="slbl">Max HR</div><div class="sval">' + str(maxhr) + '</div></div>'
    '<div class="stat-box"><div class="slbl">BMI</div><div class="sval">' + str(bmi) + '</div></div>'
    '<div class="stat-box"><div class="slbl">Oldpeak</div><div class="sval">' + str(oldpk) + '</div></div>'
    '</div>',
    unsafe_allow_html=True
)

predict_btn = st.button("🫀 PREDICT HEART DISEASE RISK", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

ui = {
    "Age":age,"Sex":sex,"ChestPainType":cp,"RestingBP":rbp,"Cholesterol":chol,
    "FastingBS":fbs,"RestingECG":ecg,"MaxHR":maxhr,"ExerciseAngina":ea,
    "Oldpeak":oldpk,"ST_Slope":slope,"BMI":bmi,"SmokingStatus":smoking,
    "Diabetes":diabetes,"BloodSugar_mg_dL":bs,"HDL_mg_dL":hdl,"LDL_mg_dL":ldl,
    "Triglycerides_mg_dL":trig,"PhysicalActivityLevel":activity,
    "AlcoholConsumption":alcohol,"FamilyHistoryHD":family_hd,
    "StressLevel":stress,"SleepHoursPerNight":sleep,
    "CreatinineLevel":creat,"CRP_mg_L":crp,
}

if predict_btn:
    inp   = prep_input(ui, scaler, sel_feat)
    pred  = best_model.predict(inp)[0]
    prob  = best_model.predict_proba(inp)[0][1]
    conf  = prob if pred == 1 else (1 - prob)

    # ── 1. Result banner ──────────────────────────────────────────────────
    if pred == 1:
        st.markdown(
            '<div class="result-high">'
            '<span style="font-size:1.4rem;">⚠️</span>'
            '<div>'
            '<div class="rtitle" style="color:#c0392b;">High Risk of Heart Disease — Confidence: '
            + f"{conf:.1%}" +
            '</div>'
            '<div class="rbody">Based on the provided parameters, the model indicates an elevated '
            'likelihood of heart disease. Please consult a cardiologist immediately.</div>'
            '</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">'
            '<span style="font-size:1.4rem;">✅</span>'
            '<div>'
            '<div class="rtitle" style="color:#1e8449;">Low Risk of Heart Disease — Confidence: '
            + f"{conf:.1%}" +
            '</div>'
            '<div class="rbody">Based on the provided parameters, the model indicates a low '
            'likelihood of heart disease. Continue maintaining a heart-healthy lifestyle.</div>'
            '</div></div>',
            unsafe_allow_html=True
        )

    # ── 2. Prediction Probability bars ───────────────────────────────────
    pd_pct  = round(prob * 100, 1)
    ph_pct  = round((1 - prob) * 100, 1)
    pd_w    = str(pd_pct) + "%"
    ph_w    = str(ph_pct) + "%"

    prob_html = (
        '<div class="info-card">'
        '<div class="info-card-title">📊 Prediction Probability</div>'
        # Disease row
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
        '<div style="width:115px;font-size:0.8rem;font-weight:600;color:#c0392b;flex-shrink:0;">Heart Disease</div>'
        '<div style="flex:1;background:#fde8e8;border-radius:20px;height:18px;overflow:hidden;">'
        '<div style="width:' + pd_w + ';background:#c0392b;height:100%;border-radius:20px;"></div>'
        '</div>'
        '<div style="width:46px;text-align:right;font-size:0.88rem;font-weight:700;color:#c0392b;flex-shrink:0;">'
        + str(pd_pct) + '%</div>'
        '</div>'
        # Healthy row
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<div style="width:115px;font-size:0.8rem;font-weight:600;color:#1e8449;flex-shrink:0;">Healthy</div>'
        '<div style="flex:1;background:#eafaf1;border-radius:20px;height:18px;overflow:hidden;">'
        '<div style="width:' + ph_w + ';background:#27ae60;height:100%;border-radius:20px;"></div>'
        '</div>'
        '<div style="width:46px;text-align:right;font-size:0.88rem;font-weight:700;color:#1e8449;flex-shrink:0;">'
        + str(ph_pct) + '%</div>'
        '</div>'
        '</div>'
    )
    st.markdown(prob_html, unsafe_allow_html=True)

    # ── 3. Risk Factor Analysis ───────────────────────────────────────────
    cards = []

    # Cholesterol
    if chol >= 240:   cards.append(rf_card("🔴","Total Cholesterol",str(chol)+" mg/dL","high","High risk (≥240 mg/dL)"))
    elif chol >= 200: cards.append(rf_card("🟡","Total Cholesterol",str(chol)+" mg/dL","medium","Borderline (200–239 mg/dL)"))
    else:             cards.append(rf_card("🟢","Total Cholesterol",str(chol)+" mg/dL","low","Desirable (<200 mg/dL)"))

    # BMI
    if bmi >= 30:     cards.append(rf_card("🔴","BMI",str(bmi),"high","Obese (≥30)"))
    elif bmi >= 25:   cards.append(rf_card("🟡","BMI",str(bmi),"medium","Overweight (25–29.9)"))
    else:             cards.append(rf_card("🟢","BMI",str(bmi),"low","Healthy (18.5–24.9)"))

    # HDL
    if hdl < 40:      cards.append(rf_card("🔴","HDL Cholesterol",str(hdl)+" mg/dL","high","Low — major risk (<40)"))
    elif hdl < 60:    cards.append(rf_card("🟡","HDL Cholesterol",str(hdl)+" mg/dL","medium","Borderline (40–59)"))
    else:             cards.append(rf_card("🟢","HDL Cholesterol",str(hdl)+" mg/dL","low","Optimal (≥60 mg/dL)"))

    # LDL
    if ldl >= 160:    cards.append(rf_card("🔴","LDL Cholesterol",str(ldl)+" mg/dL","high","High (≥160 mg/dL)"))
    elif ldl >= 130:  cards.append(rf_card("🟡","LDL Cholesterol",str(ldl)+" mg/dL","medium","Borderline (130–159)"))
    else:             cards.append(rf_card("🟢","LDL Cholesterol",str(ldl)+" mg/dL","low","Optimal (<130 mg/dL)"))

    # Resting BP
    if rbp >= 140:    cards.append(rf_card("🔴","Resting BP",str(rbp)+" mmHg","high","Stage 2 hypertension (≥140)"))
    elif rbp >= 120:  cards.append(rf_card("🟡","Resting BP",str(rbp)+" mmHg","medium","Elevated (120–139)"))
    else:             cards.append(rf_card("🟢","Resting BP",str(rbp)+" mmHg","low","Normal (<120 mmHg)"))

    # Blood Sugar
    if bs >= 126:     cards.append(rf_card("🔴","Blood Sugar",str(bs)+" mg/dL","high","Diabetic range (≥126)"))
    elif bs >= 100:   cards.append(rf_card("🟡","Blood Sugar",str(bs)+" mg/dL","medium","Pre-diabetic (100–125)"))
    else:             cards.append(rf_card("🟢","Blood Sugar",str(bs)+" mg/dL","low","Normal (70–99 mg/dL)"))

    # CRP
    if crp > 3:       cards.append(rf_card("🔴","CRP (Inflammation)",str(crp)+" mg/L","high","High cardiac risk (>3)"))
    elif crp > 1:     cards.append(rf_card("🟡","CRP (Inflammation)",str(crp)+" mg/L","medium","Moderate risk (1–3)"))
    else:             cards.append(rf_card("🟢","CRP (Inflammation)",str(crp)+" mg/L","low","Low risk (<1 mg/L)"))

    # Triglycerides
    if trig >= 200:   cards.append(rf_card("🔴","Triglycerides",str(trig)+" mg/dL","high","High (≥200 mg/dL)"))
    elif trig >= 150: cards.append(rf_card("🟡","Triglycerides",str(trig)+" mg/dL","medium","Borderline (150–199)"))
    else:             cards.append(rf_card("🟢","Triglycerides",str(trig)+" mg/dL","low","Normal (<150 mg/dL)"))

    # Smoking
    if smoking == "Current": cards.append(rf_card("🔴","Smoking","Current","high","2–4x higher HD risk"))
    elif smoking == "Former":cards.append(rf_card("🟡","Smoking","Former","medium","Reduced but still elevated"))
    else:                    cards.append(rf_card("🟢","Smoking","Never","low","No smoking risk"))

    # Exercise Angina
    if ea == "Y": cards.append(rf_card("🔴","Exercise Angina","Yes","high","Strong cardiac predictor"))
    else:         cards.append(rf_card("🟢","Exercise Angina","No","low","No exercise-induced pain"))

    # Family History
    if family_hd == 1: cards.append(rf_card("🔴","Family History HD","Yes","high","40–60% increased genetic risk"))
    else:              cards.append(rf_card("🟢","Family History HD","No","low","No genetic risk flag"))

    # Physical Activity
    if activity == "Low":      cards.append(rf_card("🔴","Physical Activity","Low","high","Major modifiable risk factor"))
    elif activity == "Moderate":cards.append(rf_card("🟡","Physical Activity","Moderate","medium","Aim for High"))
    else:                      cards.append(rf_card("🟢","Physical Activity","High","low","Excellent for heart health"))

    # Stress
    if stress == "High":     cards.append(rf_card("🔴","Stress Level","High","high","Elevates cortisol and BP"))
    elif stress == "Moderate":cards.append(rf_card("🟡","Stress Level","Moderate","medium","Manage with relaxation"))
    else:                    cards.append(rf_card("🟢","Stress Level","Low","low","Good stress management"))

    # Sleep
    if sleep < 6:   cards.append(rf_card("🔴","Sleep",str(sleep)+" hrs","high","Under 6 hrs raises HD risk 48%"))
    elif sleep > 9: cards.append(rf_card("🟡","Sleep",str(sleep)+" hrs","medium","Excess sleep also elevates risk"))
    else:           cards.append(rf_card("🟢","Sleep",str(sleep)+" hrs","low","Optimal range (7–9 hrs)"))

    grid_inner = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">' + "".join(cards) + "</div>"
    rf_html = (
        '<div class="info-card">'
        '<div class="info-card-title">🔍 Risk Factor Analysis</div>'
        + grid_inner +
        '</div>'
    )
    st.markdown(rf_html, unsafe_allow_html=True)

    # ── 4. Model Used ─────────────────────────────────────────────────────
    yp_d = best_model.predict(x_te_s)
    ypr_d= best_model.predict_proba(x_te_s)[:,1]
    m_acc = accuracy_score(y_te, yp_d)
    m_f1  = f1_score(y_te, yp_d)
    m_auc = roc_auc_score(y_te, ypr_d)

    def metric_box(val, lbl):
        return (
            '<div style="background:white;border:1px solid #e8d8d0;border-radius:8px;'
            'padding:0.5rem 0.9rem;text-align:center;min-width:70px;">'
            '<div style="font-size:1rem;font-weight:700;color:#c0392b;font-family:monospace;">'
            + val +
            '</div>'
            '<div style="font-size:0.65rem;color:#7a6058;text-transform:uppercase;'
            'letter-spacing:0.06em;margin-top:2px;">' + lbl + '</div>'
            '</div>'
        )

    model_html = (
        '<div class="info-card">'
        '<div class="info-card-title">🤖 Model Used</div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'flex-wrap:wrap;gap:12px;">'
        '<div>'
        '<div style="font-size:1.05rem;font-weight:700;color:#c0392b;">🏆 ' + best_name + '</div>'
        '<div style="font-size:0.77rem;color:#7a6058;margin-top:3px;">'
        'Best of 7 classifiers &nbsp;·&nbsp; 10,000 patients &nbsp;·&nbsp; 26 features'
        '</div>'
        '</div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        + metric_box(f"{m_acc:.3f}", "Accuracy")
        + metric_box(f"{m_f1:.3f}",  "F1 Score")
        + metric_box(f"{m_auc:.3f}", "ROC-AUC")
        +
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(model_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 5. Health Tips ────────────────────────────────────────────────────
    if pred == 0:
        st.markdown('<div class="tips-header">💚 Heart Health Tips to Maintain Low Risk</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="tips-grid">'
            '<div class="tip-card"><div class="tip-icon">🏃</div>'
            '<div class="tip-title">Stay Active</div>'
            '<div class="tip-body">Aim for 150 minutes of moderate aerobic exercise per week to keep your heart strong.</div></div>'

            '<div class="tip-card"><div class="tip-icon">🥗</div>'
            '<div class="tip-title">Heart-Healthy Diet</div>'
            '<div class="tip-body">Reduce saturated fats, sodium, and processed foods. Increase fruits, vegetables, and whole grains.</div></div>'

            '<div class="tip-card"><div class="tip-icon">🩺</div>'
            '<div class="tip-title">Regular Checkups</div>'
            '<div class="tip-body">Monitor blood pressure, cholesterol, and blood sugar levels with annual health screenings.</div></div>'

            '<div class="tip-card"><div class="tip-icon">😴</div>'
            '<div class="tip-title">Quality Sleep</div>'
            '<div class="tip-body">Poor sleep increases cardiovascular risk. Aim for 7–9 hours of quality sleep each night.</div></div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="tips-header">❤️ Immediate Steps to Reduce Your Risk</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="tips-grid">'
            '<div class="tip-card-red"><div class="tip-icon">👨‍⚕️</div>'
            '<div class="tip-title">See a Cardiologist</div>'
            '<div class="tip-body">Schedule an appointment immediately. Early intervention significantly improves outcomes.</div></div>'

            '<div class="tip-card-red"><div class="tip-icon">💊</div>'
            '<div class="tip-title">Medication Review</div>'
            '<div class="tip-body">Discuss cholesterol-lowering statins and blood pressure medications with your doctor.</div></div>'

            '<div class="tip-card-red"><div class="tip-icon">🚭</div>'
            '<div class="tip-title">Quit Smoking</div>'
            '<div class="tip-body">Stopping smoking reduces heart disease risk by 50% within just one year.</div></div>'

            '<div class="tip-card-red"><div class="tip-icon">🥦</div>'
            '<div class="tip-title">Diet and Weight</div>'
            '<div class="tip-body">A Mediterranean diet rich in omega-3s, vegetables, and lean protein significantly reduces risk.</div></div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        '<strong>⚠️ Medical Disclaimer:</strong> This tool is intended for educational and research '
        'purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. '
        'Always consult a qualified healthcare provider.'
        '</div>',
        unsafe_allow_html=True
    )

else:
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 1rem;color:#b0988e;">'
        '<div style="font-size:2.5rem;margin-bottom:0.8rem;">🫀</div>'
        '<div style="font-size:0.95rem;font-weight:500;">'
        'Fill in the patient details above and click<br>'
        '<strong style="color:#c0392b;">Predict Heart Disease Risk</strong> to see the result.'
        '</div></div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Heart Disease Prediction System &nbsp;·&nbsp; Powered by Machine Learning &nbsp;·&nbsp; Not a medical device'
    '</div>',
    unsafe_allow_html=True
)
