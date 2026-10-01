import os
import io
import re
import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

import torch
import torch.nn as nn


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Live Adversarial IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
<style>
:root {
    --bg: #0b1020;
    --bg2: #0f1630;
    --panel: #121a2e;
    --panel2: #17213a;
    --panel3: #1b2744;
    --border: #2a3a5e;
    --border-soft: #22304d;
    --text: #f7f9ff;
    --muted: #9eacc3;
    --cyan: #4cc9f0;
    --blue: #6ea8fe;
    --violet: #a78bfa;
    --mint: #4ade80;
    --amber: #fbbf24;
    --pink: #fb7185;
    --red: #ff5d73;
}

.stApp {
    background:
        radial-gradient(circle at 85% -5%, rgba(76,201,240,.16), transparent 28%),
        radial-gradient(circle at 12% 18%, rgba(167,139,250,.13), transparent 30%),
        radial-gradient(circle at 80% 80%, rgba(74,222,128,.07), transparent 24%),
        linear-gradient(145deg, #0b1020 0%, #0e1428 52%, #10182e 100%);
    color: var(--text);
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .22;
    background-image:
        linear-gradient(rgba(110,168,254,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(110,168,254,.045) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 78%);
    z-index: 0;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 50% 0%, rgba(76,201,240,.10), transparent 30%),
        linear-gradient(180deg, #10182b 0%, #0d1426 100%);
    border-right: 1px solid #293a5e;
    box-shadow: 12px 0 40px rgba(0,0,0,.18);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.15rem;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--muted) !important;
}

/* Sidebar brand */
.nav-brand {
    position: relative;
    overflow: hidden;
    padding: 17px 15px 16px;
    margin-bottom: 18px;
    border: 1px solid #2c4169;
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(76,201,240,.12), rgba(167,139,250,.10)),
        rgba(18,26,46,.78);
    box-shadow: 0 10px 28px rgba(0,0,0,.16);
}

.nav-brand:after {
    content: "";
    position: absolute;
    width: 100px;
    height: 100px;
    right: -38px;
    top: -48px;
    border-radius: 50%;
    background: rgba(76,201,240,.13);
    filter: blur(5px);
}

.nav-brand .title {
    position: relative;
    z-index: 1;
    font-size: 20px;
    font-weight: 900;
    letter-spacing: -.55px;
}

.nav-brand .sub {
    position: relative;
    z-index: 1;
    color: #a9b9d3;
    font-size: 12px;
    margin-top: 6px;
}

/* Sidebar workspace label */
[data-testid="stSidebar"] .stRadio > label {
    color: #dbe7fb !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 850;
    margin-bottom: 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px 11px !important;
    border: 1px solid transparent;
    border-radius: 11px;
    background: rgba(255,255,255,.025);
    transition: all .18s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(76,201,240,.08);
    border-color: #2c4169;
    transform: translateX(2px);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(76,201,240,.16), rgba(167,139,250,.10));
    border-color: #3a5d91;
    box-shadow: inset 3px 0 0 var(--cyan), 0 7px 20px rgba(76,201,240,.07);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 850;
}

[data-testid="stSidebar"] hr {
    border-color: #2a3855 !important;
    margin: 24px 0;
}

/* Sidebar status */
.sidebar-status {
    border: 1px solid #294a4a;
    border-radius: 14px;
    padding: 12px 13px;
    background: linear-gradient(135deg, rgba(74,222,128,.10), rgba(76,201,240,.06));
}

.sidebar-status.waiting {
    border-color: #5c4a2a;
    background: linear-gradient(135deg, rgba(251,191,36,.10), rgba(167,139,250,.05));
}

.sidebar-status .status-title {
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .6px;
}

.sidebar-status .status-detail {
    color: #9fb0c8;
    font-size: 11px;
    margin-top: 4px;
}

.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 14px currentColor;
}

.dot-green { color: var(--mint); background: var(--mint); }
.dot-red { color: var(--red); background: var(--red); }
.dot-amber { color: var(--amber); background: var(--amber); }

/* Reset */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #202b46, #252e49) !important;
    border: 1px solid #3a4d72 !important;
    color: #f7f9ff !important;
    box-shadow: 0 8px 22px rgba(0,0,0,.14);
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--cyan) !important;
    background: linear-gradient(135deg, #243653, #302d58) !important;
    box-shadow: 0 0 22px rgba(76,201,240,.13);
}

/* Main content */
.block-container {
    position: relative;
    z-index: 1;
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
}

p, li, label {
    color: #d2dced;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid #30476f;
    border-radius: 22px;
    padding: 28px 31px;
    background:
        linear-gradient(135deg, rgba(76,201,240,.13), rgba(167,139,250,.10) 48%, rgba(74,222,128,.05)),
        rgba(17,25,45,.88);
    margin-bottom: 24px;
    box-shadow: 0 18px 45px rgba(0,0,0,.14);
}

.hero:before {
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    border-radius: 50%;
    right: -90px;
    top: -120px;
    background: rgba(76,201,240,.12);
    filter: blur(5px);
}

.hero:after {
    content: "";
    position: absolute;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    right: 110px;
    bottom: -115px;
    background: rgba(167,139,250,.09);
    filter: blur(8px);
}

.hero-title {
    position: relative;
    z-index: 1;
    font-size: 31px;
    font-weight: 900;
    letter-spacing: -1px;
}

.hero-subtitle {
    position: relative;
    z-index: 1;
    color: #aebed7;
    margin-top: 7px;
    font-size: 14px;
}

/* Cards */
.card {
    background:
        linear-gradient(145deg, rgba(23,33,58,.96), rgba(15,23,42,.96));
    border: 1px solid #2a3c61;
    border-radius: 17px;
    padding: 18px;
    min-height: 112px;
    box-shadow: 0 12px 28px rgba(0,0,0,.10);
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}

.card:hover {
    transform: translateY(-2px);
    border-color: #3e5f91;
    box-shadow: 0 15px 32px rgba(76,201,240,.08);
}

.card .eyebrow {
    color: #91a4c2;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 850;
}

.card .value {
    font-size: 30px;
    font-weight: 900;
    margin-top: 7px;
    background: linear-gradient(90deg, #ffffff, #a9dfff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card .hint {
    color: #91a0b7;
    font-size: 12px;
    margin-top: 3px;
}

/* Sections */
.section-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 850;
    color: #9eb0cc;
    margin: 24px 0 10px 0;
}

.info-strip {
    background:
        linear-gradient(90deg, rgba(76,201,240,.08), rgba(167,139,250,.06));
    border: 1px solid #2d456c;
    border-radius: 13px;
    padding: 13px 15px;
    color: #c0cde0;
    font-size: 12px;
}

/* Alerts */
.alert-card {
    background: linear-gradient(135deg, #172035, #121a2d);
    border: 1px solid #2b3b5c;
    border-left: 4px solid var(--red);
    border-radius: 13px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: 0 7px 20px rgba(0,0,0,.09);
}

.alert-card.high { border-left-color: var(--amber); }
.alert-card.medium { border-left-color: var(--blue); }
.alert-card.low { border-left-color: var(--mint); }

.alert-title {
    font-weight: 800;
    font-size: 14px;
}

.alert-meta {
    color: #93a3bc;
    font-size: 11px;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    border: 1px solid #3a4b6e;
    background: #1b2741;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 800;
    margin-right: 5px;
}

.badge-red {
    color: #ff9baa;
    border-color: #78404d;
    background: #351a24;
}

.badge-green {
    color: #7cf0a9;
    border-color: #316b4b;
    background: #123020;
}

.badge-blue {
    color: #9bc7ff;
    border-color: #3b5d91;
    background: #152947;
}

/* Streamlit widgets */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #151f36, #11192c);
    border: 1px solid #2b3d62;
    padding: 14px 16px;
    border-radius: 15px;
    box-shadow: 0 9px 25px rgba(0,0,0,.10);
}

.stButton > button {
    min-height: 43px;
    border-radius: 11px;
    font-weight: 800;
    border: 1px solid #354a70;
    background: #18233b;
    color: #f5f8ff;
    transition: all .18s ease;
}

.stButton > button:hover {
    border-color: var(--cyan);
    background: #1d2b49;
    box-shadow: 0 0 22px rgba(76,201,240,.10);
    transform: translateY(-1px);
}

button[kind="primary"] {
    background: linear-gradient(135deg, #3478f6, #7c5cff) !important;
    border-color: #6b91f8 !important;
    box-shadow: 0 10px 28px rgba(76,120,246,.20) !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4388ff, #8a6dff) !important;
    box-shadow: 0 0 28px rgba(76,201,240,.18) !important;
}

/* Inputs */
[data-testid="stFileUploader"] {
    border-radius: 14px;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
textarea,
input {
    border-radius: 10px !important;
}

[data-baseweb="select"] > div {
    background: #18233b !important;
    border-color: #354a70 !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--cyan) !important;
    border-color: #b9efff !important;
}

.small-note {
    color: #8293ad;
    font-size: 11px;
    line-height: 1.5;
}

/* Expander / dataframe */
[data-testid="stExpander"] {
    background: rgba(18,26,46,.72);
    border: 1px solid #2a3c60;
    border-radius: 14px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2a3c60;
    border-radius: 13px;
    overflow: hidden;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}
::-webkit-scrollbar-track {
    background: #0b1120;
}
::-webkit-scrollbar-thumb {
    background: #314568;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #47658f;
}

/* Hide Streamlit footer */
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
def default_state():
    return {
        "trained": False,
        "preprocessor": None,
        "rf": None,
        "torch_model": None,
        "feature_names": None,
        "target_col": None,
        "training_summary": {},
        "test_predictions": None,
        "traffic_filename": None,
        "last_analysis_time": None,
        "adv_result": None,
        "adv_history": [],
        "replay_index": 0,
        "replay_active": False,
        "alert_status": {},
        "train_file_signature": None,
        "train_filename": None,
        "train_preview": None,
        "traffic_file_signature": None,
        "traffic_preview": None,
        "adv_run_id": 0,
    }


def initialize_state():
    for k, v in default_state().items():
        if k not in st.session_state:
            st.session_state[k] = v


initialize_state()


def reset_everything():
    """Hard reset: model, uploaded analysis, replay and adversarial state."""
    st.session_state.clear()
    initialize_state()


def reset_analysis_only():
    st.session_state.test_predictions = None
    st.session_state.traffic_filename = None
    st.session_state.last_analysis_time = None
    st.session_state.adv_result = None
    st.session_state.adv_history = []
    st.session_state.replay_index = 0
    st.session_state.replay_active = False
    st.session_state.alert_status = {}


# ============================================================
# ML HELPERS
# ============================================================
def find_target(df):
    candidates = [
        "label", "Label", "LABEL", "target", "Target", "class", "Class",
        "y", "attack", "Attack"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns
    }
    for key in ["label", "target", "class", "attack"]:
        if key in normalized:
            return normalized[key]
    return None


def binary_target(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s).dropna().unique().tolist())
        if set(vals).issubset({0, 1}):
            return s.astype(int), {0: "Normal", 1: "Attack"}
    low = s.astype(str).str.strip().str.lower()
    normal_words = {"normal", "benign", "0", "false", "no"}
    y = (~low.isin(normal_words)).astype(int)
    return y, {0: "Normal", 1: "Attack"}


def build_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_torch(X, y, epochs=8):
    torch.manual_seed(42)
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    return model


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32))
        ).numpy()
    return (p >= 0.5).astype(int), p


def fgsm_attack(model, X, y, epsilon):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float32)
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, yt)
    model.zero_grad()
    loss.backward()
    # Untargeted FGSM: maximize loss so an Attack prediction is pushed toward misclassification.
    adv = x + epsilon * x.grad.sign()
    return adv.detach().numpy()


def pgd_attack(model, X, y, epsilon, alpha, steps):
    model.eval()
    x0 = torch.tensor(X, dtype=torch.float32)
    x = x0.clone()
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(steps):
        x.requires_grad_(True)
        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, yt)
        model.zero_grad()
        loss.backward()
        # Untargeted PGD: gradient ascent on the classification loss.
        x = x + alpha * x.grad.sign()
        x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon).detach()
    return x.numpy()


def prepare_feature_frame(df):
    """Apply exactly the same feature-selection rules used during training."""
    target = st.session_state.target_col
    X = df.drop(columns=[target], errors="ignore").copy()
    X = X.drop(
        columns=[
            c for c in X.columns
            if str(c).lower() in {"id", "attack_cat", "attack_category"}
        ],
        errors="ignore",
    )
    for c in st.session_state.feature_names:
        if c not in X.columns:
            X[c] = np.nan
    return X[st.session_state.feature_names]


def signature_file(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()


# ============================================================
# UI HELPERS
# ============================================================
def header(title, subtitle=None, icon="🛡️"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{icon} {title}</div>
            {f'<div class="hero-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(eyebrow, value, hint=""):
    return f"""
    <div class="card">
        <div class="eyebrow">{eyebrow}</div>
        <div class="value">{value}</div>
        <div class="hint">{hint}</div>
    </div>
    """


def render_card_row(items):
    cols = st.columns(len(items))
    for col, (eyebrow, value, hint) in zip(cols, items):
        col.markdown(card(eyebrow, value, hint), unsafe_allow_html=True)


def severity(prob):
    if prob >= 0.90:
        return "CRITICAL", "red"
    if prob >= 0.70:
        return "HIGH", "high"
    if prob >= 0.50:
        return "MEDIUM", "medium"
    return "LOW", "low"


def first_existing(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def add_operational_columns(df):
    out = df.copy()
    out["severity"] = [severity(float(x))[0] if int(p) == 1 else "NORMAL"
                       for p, x in zip(out["prediction"], out["attack_probability"])]

    category_col = first_existing(out, ["attack_cat", "attack_category"])
    if category_col:
        out["display_attack_type"] = out[category_col].fillna("Unknown").astype(str)
        out.loc[out["prediction"] == 0, "display_attack_type"] = "Normal"
    else:
        out["display_attack_type"] = np.where(
            out["prediction"] == 1, "Detected Attack", "Normal"
        )
    return out


def metric_cards(y, pred, prob=None):
    acc = accuracy_score(y, pred)
    pre = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    auc = (
        roc_auc_score(y, prob)
        if prob is not None and len(np.unique(y)) == 2
        else None
    )
    render_card_row([
        ("Accuracy", f"{acc*100:.2f}%", "clean traffic"),
        ("Precision", f"{pre*100:.2f}%", "attack alerts"),
        ("Recall", f"{rec*100:.2f}%", "attacks caught"),
        ("F1 Score", f"{f1*100:.2f}%", "balanced metric"),
        ("ROC-AUC", "—" if auc is None else f"{auc*100:.2f}%", "ranking quality"),
    ])


def plot_theme(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,26,46,0.72)",
        font=dict(color="#dfe8f7"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#dfe8f7")),
    )
    fig.update_xaxes(gridcolor="#263957", zerolinecolor="#263957", linecolor="#344a70")
    fig.update_yaxes(gridcolor="#263957", zerolinecolor="#263957", linecolor="#344a70")
    return fig

def render_alert_feed(df, limit=8):
    alerts = df[df["prediction"] == 1].copy()
    if alerts.empty:
        st.markdown(
            '<div class="info-strip">No detected threats in the current traffic set.</div>',
            unsafe_allow_html=True,
        )
        return

    alerts = alerts.sort_values("attack_probability", ascending=False).head(limit)
    src_col = first_existing(alerts, ["srcip", "src_ip", "source_ip"])
    dst_col = first_existing(alerts, ["dstip", "dst_ip", "destination_ip"])
    proto_col = first_existing(alerts, ["proto", "protocol"])

    for idx, row in alerts.iterrows():
        sev, css = severity(float(row["attack_probability"]))
        src = str(row[src_col]) if src_col else "source unavailable"
        dst = str(row[dst_col]) if dst_col else "destination unavailable"
        proto = str(row[proto_col]) if proto_col else "—"
        typ = str(row.get("display_attack_type", "Detected Attack"))
        st.markdown(
            f"""
            <div class="alert-card {css}">
                <div class="alert-title">
                    <span class="badge badge-red">{sev}</span>
                    {typ}
                </div>
                <div class="alert-meta">
                    {src} → {dst} &nbsp; • &nbsp; {proto}
                    &nbsp; • &nbsp; confidence {float(row['attack_probability'])*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_source_counts(df):
    src_col = first_existing(df, ["srcip", "src_ip", "source_ip"])
    if not src_col:
        return None
    attacks = df[df["prediction"] == 1]
    if attacks.empty:
        return None
    return attacks[src_col].astype(str).value_counts().head(10).rename_axis("source").reset_index(name="alerts")


def get_category_counts(df):
    col = first_existing(df, ["attack_cat", "attack_category"])
    if not col:
        return None
    attacks = df[df["prediction"] == 1]
    if attacks.empty:
        return None
    result = attacks[col].fillna("Unknown").astype(str).value_counts().reset_index()
    result.columns = ["attack_type", "count"]
    return result


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div class="nav-brand">
        <div class="title">🛡️ Live Adversarial IDS</div>
        <div class="sub">Evasion-aware intrusion detection lab</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    [
        "Dashboard",
        "Train Model",
        "Traffic Analyzer",
        "Adversarial Lab",
        "About",
    ],
)

st.sidebar.markdown("---")

if st.session_state.trained:
    st.sidebar.markdown(
        """
        <div class="sidebar-status">
            <div class="status-title">
                <span class="status-dot dot-green"></span>IDS ONLINE
            </div>
            <div class="status-detail">Detection engine is ready for traffic analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div class="sidebar-status waiting">
            <div class="status-title">
                <span class="status-dot dot-amber"></span>MODEL NOT LOADED
            </div>
            <div class="status-detail">Train the IDS to unlock traffic analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

if st.sidebar.button("↻  Reset Workspace", use_container_width=True):
    reset_everything()
    st.rerun()

st.sidebar.caption("Reset returns the workspace to a clean starting state.")


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    header(
        "Live Security Operations Dashboard",
        "A colorful command center for traffic visibility, threat investigation and adversarial resilience testing.",
        "🛰️",
    )

    # Global action row
    c1, c2, c3 = st.columns([1.3, 1.3, 4.4])
    with c1:
        if st.button("↻ Reset Workspace", type="secondary", use_container_width=True):
            reset_everything()
            st.rerun()
    with c2:
        if st.session_state.test_predictions is not None:
            if st.button("🧹 Clear Traffic", use_container_width=True):
                reset_analysis_only()
                st.rerun()

    s = st.session_state.training_summary
    if st.session_state.trained:
        render_card_row([
            ("IDS STATUS", "ONLINE", "model loaded"),
            ("TRAINING ROWS", f"{s.get('rows', 0):,}", "labelled samples"),
            ("MODEL FEATURES", f"{s.get('features', 0):,}", "after encoding"),
            ("TRAINING ATTACKS", f"{s.get('attack', 0):,}", "class 1"),
        ])
    else:
        st.markdown(
            '<div class="info-strip"><b>IDS is offline.</b> Go to <b>Train Model</b> and load your labelled UNSW-NB15 training CSV.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if st.session_state.test_predictions is None:
        st.markdown("<div class='section-label'>No live traffic loaded</div>", unsafe_allow_html=True)
        st.info("Go to Traffic Analyzer, upload the unseen/test CSV, and the operational dashboard will populate automatically.")
        st.stop()

    df = add_operational_columns(st.session_state.test_predictions)
    total = len(df)
    attacks = int(df["prediction"].sum())
    normal = total - attacks
    attack_rate = attacks / total if total else 0
    critical = int(((df["prediction"] == 1) & (df["attack_probability"] >= .90)).sum())
    high = int(((df["prediction"] == 1) & (df["attack_probability"] >= .70) & (df["attack_probability"] < .90)).sum())

    st.markdown("<div class='section-label'>Current threat posture</div>", unsafe_allow_html=True)
    render_card_row([
        ("TRAFFIC FLOWS", f"{total:,}", st.session_state.traffic_filename or "latest analysis"),
        ("ATTACKS DETECTED", f"{attacks:,}", f"{attack_rate*100:.2f}% of traffic"),
        ("CRITICAL ALERTS", f"{critical:,}", "confidence ≥ 90%"),
        ("HIGH ALERTS", f"{high:,}", "70–89.9% confidence"),
    ])

    # Threat gauge + distribution
    left, right = st.columns([1.1, 1.9])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attack_rate * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Attack exposure"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4d5a"},
                "bgcolor": "#111923",
                "borderwidth": 1,
                "bordercolor": "#273342",
                "steps": [
                    {"range": [0, 30], "color": "#10241b"},
                    {"range": [30, 60], "color": "#2b2413"},
                    {"range": [60, 100], "color": "#291418"},
                ],
            },
        ))
        plot_theme(fig, 310)
        st.plotly_chart(fig, use_container_width=True, key="attack_gauge")

    with right:
        st.markdown("<div class='section-label'>Traffic distribution</div>", unsafe_allow_html=True)
        dist = pd.DataFrame({"State": ["Normal", "Attack"], "Count": [normal, attacks]})
        fig = px.bar(
            dist,
            x="State",
            y="Count",
            text="Count",
            color="State",
            color_discrete_map={"Normal": "#35d07f", "Attack": "#ff4d5a"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        plot_theme(fig, 310)
        st.plotly_chart(fig, use_container_width=True, key="traffic_distribution")

    # Interactive investigation controls
    st.markdown("<div class='section-label'>Threat investigation</div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns([1.1, 1.1, 2.2])
    with i1:
        min_conf = st.slider("Minimum confidence", 0.50, 0.99, 0.70, 0.01)
    with i2:
        sev_filter = st.multiselect(
            "Severity",
            ["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH", "MEDIUM"],
        )
    with i3:
        search = st.text_input("Search source / destination / attack type", placeholder="e.g. 10.0.1.24 or DoS")

    alerts = df[(df["prediction"] == 1) & (df["attack_probability"] >= min_conf)].copy()
    alerts = alerts[alerts["severity"].isin(sev_filter)]
    if search.strip():
        q = search.strip().lower()
        mask = alerts.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False)).any(axis=1)
        alerts = alerts[mask]

    st.caption(f"Showing {len(alerts):,} matching alerts.")
    show_cols = [c for c in [
        first_existing(alerts, ["srcip", "src_ip", "source_ip"]),
        first_existing(alerts, ["dstip", "dst_ip", "destination_ip"]),
        first_existing(alerts, ["proto", "protocol"]),
        "display_attack_type", "attack_probability", "severity"
    ] if c]
    if show_cols:
        table = alerts[show_cols].head(100).copy()
        if "attack_probability" in table.columns:
            table["attack_probability"] = (table["attack_probability"] * 100).round(2).astype(str) + "%"
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts match the selected filters.")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        cats = get_category_counts(df)
        if cats is not None:
            fig = px.bar(cats, x="count", y="attack_type", orientation="h", text="count")
            fig.update_traces(marker_color="#ff4d5a", textposition="outside")
            fig.update_layout(title="Detected attack categories")
            plot_theme(fig, 390)
            st.plotly_chart(fig, use_container_width=True, key="attack_categories")
        else:
            st.markdown("<div class='section-label'>Attack categories</div>", unsafe_allow_html=True)
            st.info("No attack category column was found in the uploaded traffic CSV.")
    with c2:
        src = get_source_counts(df)
        if src is not None:
            fig = px.bar(src, x="alerts", y="source", orientation="h", text="alerts")
            fig.update_traces(marker_color="#5b8ff9", textposition="outside")
            fig.update_layout(title="Top sources generating alerts")
            plot_theme(fig, 390)
            st.plotly_chart(fig, use_container_width=True, key="top_sources")
        else:
            st.markdown("<div class='section-label'>Top attacking sources</div>", unsafe_allow_html=True)
            st.info("No source-IP column was found in the uploaded traffic CSV.")

    # Replay simulator
    st.markdown("<div class='section-label'>Live traffic replay</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-strip'>Use the uploaded test set as a simulated live network feed. Each click releases the next batch of flows and updates the analyst view.</div>",
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns([1, 1, 4])
    with r1:
        batch_size = st.select_slider("Batch size", options=[10, 25, 50, 100, 250, 500], value=50)
    with r2:
        if st.button("▶ Next Batch", use_container_width=True):
            st.session_state.replay_index = min(st.session_state.replay_index + batch_size, total)
            st.rerun()
    with r3:
        processed = st.session_state.replay_index
        st.progress(processed / total if total else 0, text=f"Replay progress: {processed:,} / {total:,} flows")

    if processed:
        live = df.iloc[:processed]
        live_attacks = int(live["prediction"].sum())
        render_card_row([
            ("REPLAYED", f"{processed:,}", "flows processed"),
            ("LIVE ATTACKS", f"{live_attacks:,}", "detected so far"),
            ("LIVE ATTACK RATE", f"{live_attacks/processed*100:.2f}%", "running rate"),
        ])
        st.markdown("<div class='section-label'>Latest alert feed</div>", unsafe_allow_html=True)
        render_alert_feed(live, limit=7)


# ============================================================
# TRAIN MODEL
# ============================================================
# TRAIN MODEL
# ============================================================
elif page == "Train Model":
    header(
        "Train the IDS",
        "Build the detection engine from labelled network-flow data. Your trained model stays loaded while you move between workspace tabs.",
        "🧠",
    )

    train_file = st.file_uploader(
        "Upload labelled training CSV",
        type=["csv"],
        key="train_upload",
    )
    epochs = st.slider("Neural-network epochs", 3, 30, 8)

    # Process a newly uploaded file only once.
    if train_file is not None:
        sig = signature_file(train_file)
        if sig != st.session_state.train_file_signature:
            df = pd.read_csv(train_file)
            st.session_state.train_file_signature = sig
            st.session_state.train_filename = train_file.name
            st.session_state.train_preview = df.head(100).copy()

            target = find_target(df)
            st.session_state.training_summary["uploaded_rows"] = len(df)
            st.session_state.training_summary["uploaded_columns"] = len(df.columns)
            st.session_state.training_summary["uploaded_target"] = target

    summary = st.session_state.training_summary
    target = summary.get("uploaded_target")

    if st.session_state.train_filename:
        st.markdown(
            f"<div class='info-strip'>📁 <b>Loaded training dataset:</b> {st.session_state.train_filename} &nbsp; • &nbsp; "
            f"{summary.get('uploaded_rows', 0):,} rows &nbsp; • &nbsp; "
            f"{summary.get('uploaded_columns', 0):,} columns</div>",
            unsafe_allow_html=True,
        )

    # If the uploader disappears during a page switch, the stored model/data summary
    # remains visible instead of looking like everything was lost.
    if st.session_state.trained:
        st.markdown(
            f"<div class='sidebar-status' style='margin-top:12px;'>"
            f"<div class='status-title'><span class='status-dot dot-green'></span>MODEL LOADED</div>"
            f"<div class='status-detail'>Trained from {st.session_state.train_filename or 'the uploaded dataset'} · "
            f"{st.session_state.training_summary.get('rows', 0):,} training rows</div></div>",
            unsafe_allow_html=True,
        )

    if target is None and st.session_state.trained:
        target = st.session_state.target_col

    if target is not None and st.session_state.train_preview is not None:
        preview = st.session_state.train_preview
        render_card_row([
            ("ROWS", f"{summary.get('uploaded_rows', len(preview)): ,}".replace(" ", ""), "training records"),
            ("COLUMNS", f"{summary.get('uploaded_columns', len(preview.columns)): ,}".replace(" ", ""), "raw features"),
            ("TARGET", target, "label column"),
        ])

        y_preview, _ = binary_target(preview[target])
        # Preview is only the first 100 rows, so do not call these counts "dataset totals".
        p1, p2 = st.columns(2)
        with p1:
            st.success(f"Target detected: `{target}`")
            st.write(f"Preview — Normal: **{int((y_preview == 0).sum()):,}**")
            st.write(f"Preview — Attack: **{int((y_preview == 1).sum()):,}**")
        with p2:
            dist = pd.DataFrame({
                "Class": ["Normal", "Attack"],
                "Count": [int((y_preview == 0).sum()), int((y_preview == 1).sum())]
            })
            fig = px.pie(
                dist, names="Class", values="Count", hole=.55,
                color="Class",
                color_discrete_map={"Normal":"#35d07f","Attack":"#ff4d5a"}
            )
            plot_theme(fig, 250)
            st.plotly_chart(fig, use_container_width=True, key="train_distribution")

        with st.expander("👀 Preview loaded dataset", expanded=False):
            st.dataframe(preview, use_container_width=True, hide_index=True)

        if st.button("🚀 Train / Replace IDS Model", type="primary", use_container_width=True):
            # Re-read the uploaded file from the current widget. Training itself happens once.
            df = pd.read_csv(train_file) if train_file is not None else None
            if df is None:
                st.warning("The training file is no longer attached to this page. Upload it again to retrain. The existing trained model is still available.")
            else:
                with st.spinner("Preprocessing data and training Random Forest + neural IDS..."):
                    y, _ = binary_target(df[target])
                    X = df.drop(columns=[target]).copy()
                    drop_cols = [
                        c for c in X.columns
                        if str(c).lower() in {"id", "attack_cat", "attack_category"}
                    ]
                    X = X.drop(columns=drop_cols, errors="ignore")

                    pre = build_preprocessor(X)
                    Xp = np.asarray(pre.fit_transform(X), dtype=np.float32)

                    rf = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    )
                    rf.fit(Xp, y)
                    nn_model = train_torch(Xp, y, epochs=epochs)

                    st.session_state.preprocessor = pre
                    st.session_state.rf = rf
                    st.session_state.torch_model = nn_model
                    st.session_state.feature_names = X.columns.tolist()
                    st.session_state.target_col = target
                    st.session_state.trained = True
                    st.session_state.training_summary = {
                        "rows": len(df),
                        "features": Xp.shape[1],
                        "normal": int((y == 0).sum()),
                        "attack": int((y == 1).sum()),
                        "trained_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                        "uploaded_rows": len(df),
                        "uploaded_columns": len(df.columns),
                        "uploaded_target": target,
                    }
                    st.session_state.train_filename = train_file.name
                    st.session_state.train_file_signature = signature_file(train_file)
                    st.session_state.train_preview = df.head(100).copy()
                    reset_analysis_only()

                st.success("IDS trained successfully. You can now analyze unseen traffic.")
                pred = rf.predict(Xp)
                prob = rf.predict_proba(Xp)[:, 1]
                metric_cards(y, pred, prob)

    elif not st.session_state.trained:
        st.info("Upload your labelled training CSV above to inspect it and train the IDS.")
    else:
        st.info("Your trained IDS is still loaded. Upload a new training CSV only if you want to replace it.")


# ============================================================
# TRAFFIC ANALYZER
# ============================================================
elif page == "Traffic Analyzer":
    header(
        "Traffic Analyzer",
        "Upload unseen network flows and investigate what the trained IDS detects. Analysis stays available when you switch tabs.",
        "🔎",
    )

    if not st.session_state.trained:
        st.warning("Train the IDS first.")
    else:
        f = st.file_uploader(
            "Upload unseen network-traffic CSV",
            type=["csv"],
            key="traffic_upload",
        )

        # Process a new upload once. Do not require the uploader to remain visible
        # for the stored analysis to be rendered.
        if f is not None:
            sig = signature_file(f)
            if sig != st.session_state.traffic_file_signature:
                df_new = pd.read_csv(f)

                with st.spinner("Running traffic through the IDS..."):
                    X = prepare_feature_frame(df_new)
                    Xp = np.asarray(
                        st.session_state.preprocessor.transform(X),
                        dtype=np.float32
                    )
                    pred = st.session_state.rf.predict(Xp)
                    prob = st.session_state.rf.predict_proba(Xp)[:, 1]

                out = df_new.copy()
                out["prediction"] = pred
                out["attack_probability"] = prob
                out["class_name"] = np.where(pred == 1, "Attack", "Normal")
                out = add_operational_columns(out)

                st.session_state.test_predictions = out
                st.session_state.traffic_filename = f.name
                st.session_state.traffic_file_signature = sig
                st.session_state.traffic_preview = df_new.head(100).copy()
                st.session_state.last_analysis_time = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                st.session_state.adv_result = None
                st.session_state.adv_history = []
                st.session_state.replay_index = 0
                st.session_state.replay_active = False
                st.session_state.alert_status = {}
                st.session_state.adv_run_id = 0

        out = st.session_state.test_predictions

        if out is None:
            st.info("Upload a CSV above. Once analyzed, the results will remain available while you move between Dashboard, Traffic Analyzer and Adversarial Lab.")
        else:
            attacks = int(out["prediction"].sum())
            normal = len(out) - attacks

            st.markdown(
                f"<div class='info-strip'>📡 <b>Active traffic dataset:</b> {st.session_state.traffic_filename or 'uploaded traffic'} "
                f"&nbsp; • &nbsp; Analysis completed {st.session_state.last_analysis_time or '—'}</div>",
                unsafe_allow_html=True,
            )

            render_card_row([
                ("TRAFFIC FLOWS", f"{len(out):,}", st.session_state.traffic_filename or "loaded dataset"),
                ("NORMAL", f"{normal:,}", f"{normal/len(out)*100:.2f}%" if len(out) else "—"),
                ("ATTACKS", f"{attacks:,}", f"{attacks/len(out)*100:.2f}%" if len(out) else "—"),
                ("ANALYZED", st.session_state.last_analysis_time or "—", "latest run"),
            ])

            if st.button("🧹 Clear This Traffic Dataset", use_container_width=True):
                reset_analysis_only()
                st.rerun()

            tab1, tab2, tab3 = st.tabs(["Overview", "Investigate", "Ground Truth"])

            with tab1:
                left, right = st.columns(2)
                with left:
                    dist = pd.DataFrame({
                        "Class": ["Normal", "Attack"],
                        "Count": [normal, attacks]
                    })
                    fig = px.pie(
                        dist, names="Class", values="Count", hole=.58,
                        color="Class",
                        color_discrete_map={"Normal":"#35d07f","Attack":"#ff4d5a"}
                    )
                    fig.update_traces(textinfo="percent+label")
                    plot_theme(fig, 390)
                    st.plotly_chart(fig, use_container_width=True, key="traffic_dist")

                with right:
                    hist = px.histogram(
                        out, x="attack_probability", nbins=25,
                        title="Attack-confidence distribution"
                    )
                    hist.update_traces(marker_color="#5b8ff9")
                    hist.update_xaxes(title="Attack probability")
                    hist.update_yaxes(title="Flows")
                    plot_theme(hist, 390)
                    st.plotly_chart(hist, use_container_width=True, key="confidence_hist")

                st.markdown("<div class='section-label'>Latest alert feed</div>", unsafe_allow_html=True)
                render_alert_feed(out, limit=8)

            with tab2:
                render_card_row([
                    ("CRITICAL", f"{int(((out.prediction==1)&(out.attack_probability>=.90)).sum()):,}", "≥ 90% confidence"),
                    ("HIGH", f"{int(((out.prediction==1)&(out.attack_probability.between(.70,.899999))).sum()):,}", "70–89.9%"),
                    ("MEDIUM", f"{int(((out.prediction==1)&(out.attack_probability.between(.50,.699999))).sum()):,}", "50–69.9%"),
                ])
                st.dataframe(
                    out.sort_values("attack_probability", ascending=False).head(500),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "⬇️ Download Full Predictions",
                    out.to_csv(index=False).encode("utf-8"),
                    "ids_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with tab3:
                target = st.session_state.target_col
                if target not in out.columns:
                    st.info("No ground-truth label was found in this file, so this upload is being treated as unseen traffic.")
                else:
                    y, _ = binary_target(out[target])
                    metric_cards(y, out["prediction"].values, out["attack_probability"].values)
                    cm = confusion_matrix(y, out["prediction"])
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        x=["Normal", "Attack"],
                        y=["Normal", "Attack"],
                        labels={"x":"Predicted", "y":"Actual"},
                        color_continuous_scale=[[0,"#0f1721"],[1,"#5b8ff9"]],
                    )
                    fig.update_layout(title="Confusion Matrix")
                    plot_theme(fig, 430)
                    st.plotly_chart(fig, use_container_width=True, key="analyzer_cm")


# ============================================================
elif page == "Adversarial Lab":
    header(
        "Adversarial Attack Lab",
        "Stress-test the differentiable neural IDS and measure Attack → Normal evasion in feature space.",
        "⚡",
    )

    # Reset controls specifically for the lab
    r1, r2, r3 = st.columns([1.2, 1.2, 4.6])
    with r1:
        if st.button("↻ Reset Lab", use_container_width=True):
            st.session_state.adv_result = None
            st.session_state.adv_history = []
            st.rerun()
    with r2:
        if st.button("↻ Reset Everything", use_container_width=True):
            reset_everything()
            st.rerun()

    if not st.session_state.trained:
        st.warning("Train the model first.")
    elif st.session_state.test_predictions is None:
        st.info("Analyze a CSV in Traffic Analyzer first. The lab uses currently detected attack flows as its starting point.")
    else:
        source = st.session_state.test_predictions
        attacks = source[source["prediction"] == 1].copy()

        render_card_row([
            ("ATTACK CANDIDATES", f"{len(attacks):,}", "currently detected"),
            ("MODEL", "NEURAL IDS", "differentiable target"),
            ("OBJECTIVE", "ATTACK → NORMAL", "evasion test"),
        ])

        if attacks.empty:
            st.success("No predicted attacks are available for the adversarial experiment.")
        else:
            st.markdown("<div class='section-label'>Experiment controls</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='small-note'>Each run uses a fresh attack batch when enough candidates are available. "
                "FGSM/PGD use untargeted gradient ascent, so the experiment actively searches for Attack → Normal evasion.</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                n = st.slider("Attack samples", 10, min(1000, len(attacks)), min(100, len(attacks)))
            with c2:
                attack_type = st.selectbox("Attack method", ["FGSM", "PGD"])
            with c3:
                epsilon = st.slider("Perturbation strength (ε)", 0.001, 0.10, 0.01, 0.001)

            if attack_type == "PGD":
                p1, p2 = st.columns(2)
                with p1:
                    steps = st.slider("PGD steps", 2, 20, 8)
                with p2:
                    alpha = st.slider("PGD step size", 0.0005, float(epsilon), min(0.005, float(epsilon)), 0.0005)
            else:
                steps = 1
                alpha = epsilon

            st.markdown(
                f"<div class='info-strip'><b>{attack_type}</b> will test {n:,} detected attack flows with ε = {epsilon:.3f}. The experiment is performed on standardized model features.</div>",
                unsafe_allow_html=True,
            )

            if st.button("⚡ Run Adversarial Experiment", type="primary", use_container_width=True):
                with st.spinner("Generating adversarial feature-space examples and evaluating the neural IDS..."):
                    st.session_state.adv_run_id += 1
                    run_seed = int(time.time_ns() % (2**32 - 1))
                    sample = attacks.sample(
                        n=n,
                        random_state=run_seed if n < len(attacks) else None,
                    )
                    X = prepare_feature_frame(sample)
                    Xp = np.asarray(st.session_state.preprocessor.transform(X), dtype=np.float32)

                    base_pred, base_prob = predict_torch(st.session_state.torch_model, Xp)
                    mask = base_pred == 1

                    if mask.sum() == 0:
                        st.error("The neural IDS did not classify the selected records as attacks. Increase the sample size or choose another traffic file.")
                    else:
                        Xa = Xp[mask]
                        ya = np.ones(mask.sum(), dtype=np.int64)
                        if attack_type == "FGSM":
                            adv = fgsm_attack(st.session_state.torch_model, Xa, ya, epsilon)
                        else:
                            adv = pgd_attack(st.session_state.torch_model, Xa, ya, epsilon, alpha, steps)

                        adv_pred, adv_prob = predict_torch(st.session_state.torch_model, adv)
                        fooled = int((adv_pred == 0).sum())
                        total_adv = len(adv_pred)
                        asr = fooled / total_adv if total_adv else 0

                        result = {
                            "method": attack_type,
                            "epsilon": epsilon,
                            "steps": steps,
                            "tested": total_adv,
                            "evasions": fooled,
                            "asr": asr,
                            "before_attack": int((base_pred[mask] == 1).sum()),
                            "after_attack": int((adv_pred == 1).sum()),
                            "after_normal": int((adv_pred == 0).sum()),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "run_id": st.session_state.adv_run_id,
                        }
                        st.session_state.adv_result = result
                        st.session_state.adv_history.append(result)

            result = st.session_state.adv_result
            if result:
                st.markdown("<div class='section-label'>Experiment result</div>", unsafe_allow_html=True)
                render_card_row([
                    ("ATTACKS TESTED", f"{result['tested']:,}", result["method"]),
                    ("EVASIONS", f"{result['evasions']:,}", "changed to Normal"),
                    ("ATTACK SUCCESS RATE", f"{result['asr']*100:.2f}%", "higher = more successful evasion"),
                    ("EPSILON", f"{result['epsilon']:.3f}", "feature-space strength"),
                ])

                # Visual before/after
                comp = pd.DataFrame({
                    "State": ["Before perturbation", "After perturbation"],
                    "Attack": [result["before_attack"], result["after_attack"]],
                    "Normal": [0, result["after_normal"]],
                })
                fig = px.bar(
                    comp,
                    x="State",
                    y=["Attack", "Normal"],
                    barmode="stack",
                    title=f"{result['method']} — detection shift",
                    color_discrete_map={"Attack":"#ff4d5a", "Normal":"#35d07f"},
                )
                plot_theme(fig, 380)
                st.plotly_chart(fig, use_container_width=True, key="adv_shift")

                # ASR gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["asr"] * 100,
                    number={"suffix": "%"},
                    title={"text": "Evasion success rate"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ff4d5a"},
                        "bgcolor": "#111923",
                        "bordercolor": "#273342",
                    },
                ))
                plot_theme(fig, 300)
                st.plotly_chart(fig, use_container_width=True, key="adv_gauge")

                if len(st.session_state.adv_history) > 1:
                    hist = pd.DataFrame(st.session_state.adv_history)
                    fig = px.line(
                        hist,
                        x="timestamp",
                        y=hist["asr"] * 100,
                        markers=True,
                        title="Experiment history — ASR",
                    )
                    fig.update_yaxes(title="ASR (%)")
                    fig.update_traces(line_color="#9b7bff")
                    plot_theme(fig, 300)
                    st.plotly_chart(fig, use_container_width=True, key="adv_history")

                st.warning(
                    "These are feature-space adversarial perturbations for robustness research. They are not guaranteed to correspond to valid modified packets or executable attacks."
                )


# ============================================================
# ABOUT
# ============================================================
else:
    header(
        "About the Project",
        "Adversarially robust intrusion detection using machine learning on network-flow data.",
        "📘",
    )

    st.markdown(
        """
        <div class="info-strip">
        <b>Project workflow:</b> train → analyze unseen traffic → investigate alerts → replay traffic → test adversarial evasion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What this application demonstrates")
    st.markdown(
        """
        - Binary intrusion detection: <b>Normal</b> vs <b>Attack</b>.
        - Numerical and categorical preprocessing with a reusable pipeline.
        - Random Forest detection model.
        - Differentiable PyTorch neural IDS for gradient-based robustness testing.
        - FGSM and PGD feature-space adversarial experiments.
        - Confidence-based operational alert severity.
        - Source and attack-category investigation when those fields exist.
        - CSV replay to simulate a live network feed for demonstration.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset guidance")
    st.write("For UNSW-NB15, use `label` as the binary target: 0 = Normal and 1 = Attack. `attack_cat` is excluded from model inputs to avoid target leakage but can be displayed for investigation when present in an uploaded file.")

    st.markdown("### Important limitation")
    st.warning("The live replay is a demonstration of an operational IDS workflow, not a packet-capture engine. FGSM/PGD operate on standardized feature vectors and do not guarantee packet-valid network attacks.")
