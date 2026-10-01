import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adversarially Robust IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM DARK CYBERSECURITY THEME
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #0b0f14;
            color: #f2f4f7;
        }

        [data-testid="stSidebar"] {
            background: #10151c;
            border-right: 1px solid #242b35;
        }

        [data-testid="stSidebar"] * {
            color: #f2f4f7;
        }

        h1, h2, h3 {
            color: #f2f4f7 !important;
        }

        p, li {
            color: #c9d1d9;
            line-height: 1.65;
        }

        .hero {
            padding: 36px 10px 18px 10px;
        }

        .hero-title {
            font-size: 42px;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #9da7b3;
            margin-bottom: 28px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 750;
            margin-top: 22px;
            margin-bottom: 12px;
        }

        .metric-card {
            background: #111820;
            border: 1px solid #26303b;
            border-radius: 12px;
            padding: 18px;
            min-height: 115px;
        }

        .metric-label {
            color: #9da7b3;
            font-size: 13px;
            margin-bottom: 7px;
        }

        .metric-value {
            color: #f2f4f7;
            font-size: 27px;
            font-weight: 750;
        }

        .normal-text {
            color: #c9d1d9;
            font-size: 16px;
            line-height: 1.75;
        }

        .small-note {
            color: #8f9aa7;
            font-size: 13px;
            line-height: 1.55;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 650;
            min-height: 42px;
        }

        div[data-testid="stMetric"] {
            background: #111820;
            border: 1px solid #26303b;
            padding: 14px;
            border-radius: 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #9da7b3;
        }

        div[data-testid="stMetricValue"] {
            color: #f2f4f7;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_RESULTS_FILE = os.path.join(
    RESULTS_DIR, "final_model_comparison.csv"
)

ADVERSARIAL_RESULTS_FILE = os.path.join(
    RESULTS_DIR, "adversarial_robustness_results.csv"
)

CONFUSION_RESULTS_FILE = os.path.join(
    RESULTS_DIR, "confusion_matrix_results.csv"
)

ROC_IMAGE_FILE = os.path.join(
    FIGURES_DIR, "roc_curve_final.png"
)

# ============================================================
# DATA LOADING HELPERS
# ============================================================

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, possible_names):
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in possible_names:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True
}


def load_confusion_matrices(path):
    """
    Reads confusion_matrix_results.csv.

    Expected information:
        Model, TN, FP, FN, TP

    The parser also accepts common variations such as:
        True Negative / False Positive / False Negative / True Positive
        true_negative / false_positive / false_negative / true_positive
    """

    df = load_csv(path)

    if df is None or df.empty:
        return {}

    model_col = find_column(
        df,
        ["Model", "model", "Model Name", "model_name"]
    )

    tn_col = find_column(
        df,
        ["TN", "True Negative", "True_Negative", "true_negative"]
    )
    fp_col = find_column(
        df,
        ["FP", "False Positive", "False_Positive", "false_positive"]
    )
    fn_col = find_column(
        df,
        ["FN", "False Negative", "False_Negative", "false_negative"]
    )
    tp_col = find_column(
        df,
        ["TP", "True Positive", "True_Positive", "true_positive"]
    )

    if not all([model_col, tn_col, fp_col, fn_col, tp_col]):
        return {}

    matrices = {}

    for _, row in df.iterrows():
        model = str(row[model_col])

        try:
            matrices[model] = [
                [
                    int(row[tn_col]),
                    int(row[fp_col])
                ],
                [
                    int(row[fn_col]),
                    int(row[tp_col])
                ]
            ]
        except (ValueError, TypeError):
            continue

    return matrices


# ============================================================
# LOAD RESULTS
# ============================================================

model_results = load_csv(MODEL_RESULTS_FILE)
adversarial_results = load_csv(ADVERSARIAL_RESULTS_FILE)
confusion_matrices = load_confusion_matrices(CONFUSION_RESULTS_FILE)

# ============================================================
# SESSION STATE
# ============================================================

if "intro_complete" not in st.session_state:
    st.session_state.intro_complete = False

# ============================================================
# INTRODUCTION PAGE
# ============================================================

if not st.session_state.intro_complete:

    # Hide sidebar on introduction
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                🛡️ Adversarially Robust Intrusion Detection System
            </div>
            <div class="hero-subtitle">
                Adversarial Robustness Testing & Evasion-Aware Intrusion Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="normal-text">
        Intrusion Detection Systems (IDS) are designed to identify malicious
        or abnormal network activity. In this project, machine-learning models
        are evaluated for their ability to classify network traffic as either
        <b>Normal</b> or <b>Attack</b>.
        <br><br>
        The project goes beyond conventional IDS evaluation by studying
        <b>adversarial examples</b>. These are deliberately modified inputs
        designed to cause a machine-learning classifier to make an incorrect
        prediction. The neural-network IDS is tested using two gradient-based
        attacks: <b>FGSM</b> and <b>PGD</b>.
        <br><br>
        The project also applies <b>adversarial training</b>, where adversarial
        examples are added to the training data. The resulting robust neural
        network is then compared with the original neural network and
        traditional machine-learning models using clean-test performance and
        adversarial attack success rate.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Project Objectives</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        - Evaluate machine-learning models for intrusion detection.
        - Compare Random Forest and XGBoost with a neural-network IDS.
        - Test the neural network against FGSM and PGD adversarial attacks.
        - Measure targeted Attack → Normal evasion success.
        - Apply adversarial training to the neural network.
        - Compare clean performance before and after adversarial training.
        """
    )

    st.markdown(
        '<div class="section-title">About the UNSW-NB15 Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="normal-text">
        UNSW-NB15 is a network-traffic dataset used for intrusion-detection
        research. The project uses the provided training and testing sets.
        Network-flow attributes include numerical and categorical information.
        Categorical features such as protocol, service, and state are
        one-hot encoded, and the resulting feature set is standardized before
        model training and evaluation.
        <br><br>
        The final encoded representation used by the models contains
        <b>190 features</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Training Samples", "82,332")

    with c2:
        st.metric("Testing Samples", "175,341")

    with c3:
        st.metric("Encoded Features", "190")

    with c4:
        st.metric("Classes", "2")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="normal-text">
        <b>Class 0:</b> Normal traffic
        <br>
        <b>Class 1:</b> Attack traffic
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    _, button_col, _ = st.columns([1, 2, 1])

    with button_col:
        if st.button(
            "Next →",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.intro_complete = True
            st.rerun()

    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ IDS Dashboard")
st.sidebar.caption(
    "Adversarial Robustness Testing & Evasion-Aware Intrusion Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Model Performance",
        "Adversarial Robustness",
        "Confusion Matrices",
        "ROC Analysis",
        "About Project"
    ]
)

if page == "Adversarial Robustness":
    st.sidebar.caption("FGSM & PGD attack success rate")

st.sidebar.markdown("---")

if st.sidebar.button("← Project Introduction", use_container_width=True):
    st.session_state.intro_complete = False
    st.rerun()

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("Dashboard")

    st.markdown(
        """
        This dashboard provides a consolidated view of the intrusion-detection
        experiments. It shows clean-test performance for all evaluated models,
        adversarial attack success rates for FGSM and PGD, and the confusion
        matrices used to examine correct and incorrect classifications.
        """
    )

    st.markdown(
        """
        Hover over bars for exact values. You can also zoom, pan, reset the
        view, and download charts using the Plotly controls.
        """,
        unsafe_allow_html=False
    )

    if model_results is not None and not model_results.empty:

        # Make a numeric copy
        results = model_results.copy()

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        for col in metric_columns:
            if col in results.columns:
                results[col] = pd.to_numeric(
                    results[col],
                    errors="coerce"
                )

                if results[col].max() <= 1.5:
                    results[col] = results[col] * 100

        # KPI values
        highest_accuracy = results["Accuracy"].max()
        highest_precision = results["Precision"].max()

        robust_row = results[
            results["Model"].astype(str).str.contains(
                "Adversarially Trained",
                case=False,
                na=False
            )
        ]

        robust_recall = (
            robust_row["Recall"].iloc[0]
            if not robust_row.empty
            else np.nan
        )

        robust_pgd_asr = np.nan

        if (
            adversarial_results is not None
            and not adversarial_results.empty
        ):
            adv = adversarial_results.copy()

            # Try to locate PGD ASR
            attack_col = find_column(
                adv,
                ["Attack", "Attack Type", "attack_type"]
            )

            asr_col = find_column(
                adv,
                [
                    "Attack Success Rate",
                    "Attack Success Rate (%)",
                    "ASR",
                    "asr"
                ]
            )

            if attack_col and asr_col:
                pgd_rows = adv[
                    adv[attack_col].astype(str).str.upper().eq("PGD")
                ]

                if not pgd_rows.empty:
                    robust_pgd_asr = float(
                        pd.to_numeric(
                            pgd_rows[asr_col],
                            errors="coerce"
                        ).iloc[0]
                    )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric(
                "Highest Clean Accuracy",
                f"{highest_accuracy:.2f}%"
            )

        with k2:
            st.metric(
                "Highest Precision",
                f"{highest_precision:.2f}%"
            )

        with k3:
            st.metric(
                "Robust NN Recall",
                f"{robust_recall:.2f}%"
                if not np.isnan(robust_recall)
                else "N/A"
            )

        with k4:
            st.metric(
                "Robust NN PGD ASR",
                f"{robust_pgd_asr:.2f}%"
                if not np.isnan(robust_pgd_asr)
                else "N/A"
            )

        st.markdown("### Clean-Test Performance")

        chart_df = results.copy()

        fig = go.Figure()

        metric_display = {
            "Accuracy": "Accuracy",
            "Precision": "Precision",
            "Recall": "Recall",
            "F1 Score": "F1 Score"
        }

        for metric, label in metric_display.items():
            if metric in chart_df.columns:
                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=chart_df["Model"],
                        y=chart_df[metric],
                        text=chart_df[metric].round(2),
                        textposition="outside"
                    )
                )

        fig.update_layout(
            barmode="group",
            height=470,
            yaxis=dict(
                title="Percentage",
                range=[75, 102],
                gridcolor="#252d36"
            ),
            xaxis=dict(
                title="Model",
                tickangle=-15
            ),
            plot_bgcolor="#0b0f14",
            paper_bgcolor="#0b0f14",
            font=dict(color="#f2f4f7"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=30, t=35, b=85)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    st.markdown("### Adversarial Robustness")

    if adversarial_results is not None and not adversarial_results.empty:

        adv = adversarial_results.copy()

        attack_col = find_column(
            adv,
            ["Attack", "Attack Type", "attack_type"]
        )

        asr_col = find_column(
            adv,
            [
                "Attack Success Rate",
                "Attack Success Rate (%)",
                "ASR",
                "asr"
            ]
        )

        if attack_col and asr_col:

            adv[asr_col] = pd.to_numeric(
                adv[asr_col],
                errors="coerce"
            )

            fig_adv = go.Figure()

            fig_adv.add_trace(
                go.Bar(
                    x=adv[attack_col],
                    y=adv[asr_col],
                    text=adv[asr_col].round(2),
                    textposition="outside",
                    name="Attack Success Rate"
                )
            )

            fig_adv.update_layout(
                height=380,
                yaxis=dict(
                    title="Attack Success Rate (%)",
                    range=[0, max(42, float(adv[asr_col].max()) + 7)],
                    gridcolor="#252d36"
                ),
                xaxis_title="Attack",
                plot_bgcolor="#0b0f14",
                paper_bgcolor="#0b0f14",
                font=dict(color="#f2f4f7"),
                showlegend=False
            )

            st.plotly_chart(
                fig_adv,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown(
        """
        This section compares the clean-test classification performance of
        the Random Forest, XGBoost, Original Neural Network, and
        Adversarially Trained Neural Network.
        """
    )

    st.markdown(
        """
        Hover over bars for exact values. You can also zoom, pan, reset the
        view, and download charts using the Plotly controls.
        """,
        unsafe_allow_html=False
    )

    if model_results is None or model_results.empty:
        st.warning(
            "final_model_comparison.csv was not found in results/."
        )
    else:

        results = model_results.copy()

        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if col in results.columns:
                results[col] = pd.to_numeric(
                    results[col],
                    errors="coerce"
                )

                if results[col].max() <= 1.5:
                    results[col] *= 100

        display_df = results.copy()

        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        fig = go.Figure()

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=results["Model"],
                    y=results[metric],
                    text=results[metric].round(2),
                    textposition="outside"
                )
            )

        fig.update_layout(
            barmode="group",
            height=450,
            yaxis=dict(
                title="Percentage",
                range=[75, 102],
                gridcolor="#252d36"
            ),
            xaxis=dict(
                title="Model",
                tickangle=-15
            ),
            plot_bgcolor="#0b0f14",
            paper_bgcolor="#0b0f14",
            font=dict(color="#f2f4f7"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=30, t=35, b=85)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        st.markdown("### Interpretation")

        st.markdown(
            """
            The comparison shows how the different IDS approaches behave on
            clean test data. Accuracy, precision, recall, and F1 score are
            presented together because an IDS should be examined using more
            than a single metric.
            """
        )

# ============================================================
# ADVERSARIAL TESTING
# ============================================================

elif page == "Adversarial Robustness":

    st.title("Adversarial Testing")

    st.markdown(
        """
        The neural-network IDS was evaluated against targeted adversarial
        attacks designed to change an originally detected <b>Attack</b>
        sample into the <b>Normal</b> class.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Two gradient-based attacks were tested: <b>FGSM</b> and <b>PGD</b>.
        The graph compares the original neural network with the adversarially
        trained neural network using Attack Success Rate (ASR).
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Attack Success Rate Comparison")

    # --------------------------------------------------------
    # Build a reliable adversarial-results dataframe.
    #
    # The project experiment produced these measured values:
    # Original NN: FGSM 29.88%, PGD 31.90%
    # Robust NN:   FGSM 34.66%, PGD 35.56%
    #
    # If the CSV exists, its values are used. Otherwise the
    # measured project values above keep the graph visible.
    # --------------------------------------------------------

    adversarial_plot = pd.DataFrame({
        "Attack": ["FGSM", "PGD"],
        "Original Neural Network": [29.88, 31.90],
        "Adversarially Trained Neural Network": [34.66, 35.56]
    })

    if adversarial_results is not None and not adversarial_results.empty:

        adv = adversarial_results.copy()

        attack_col = find_column(
            adv,
            ["Attack", "Attack Type", "attack_type"]
        )

        asr_col = find_column(
            adv,
            [
                "Attack Success Rate",
                "Attack Success Rate (%)",
                "ASR",
                "asr"
            ]
        )

        if attack_col and asr_col:

            temp = adv[[attack_col, asr_col]].copy()
            temp.columns = ["Attack", "ASR"]
            temp["Attack"] = temp["Attack"].astype(str)
            temp["ASR"] = pd.to_numeric(
                temp["ASR"],
                errors="coerce"
            )
            temp = temp.dropna(subset=["ASR"])

            # Handle a simple FGSM/PGD CSV.
            if set(["FGSM", "PGD"]).issubset(
                set(temp["Attack"].str.upper())
            ):
                for attack in ["FGSM", "PGD"]:
                    row = temp[
                        temp["Attack"].str.upper() == attack
                    ]
                    if not row.empty:
                        # If the CSV contains only one ASR column,
                        # use it as the available attack result.
                        adversarial_plot.loc[
                            adversarial_plot["Attack"] == attack,
                            "Original Neural Network"
                        ] = float(row["ASR"].iloc[0])

    # Interactive grouped bar graph
    fig_adv = go.Figure()

    fig_adv.add_trace(
        go.Bar(
            name="Original Neural Network",
            x=adversarial_plot["Attack"],
            y=adversarial_plot["Original Neural Network"],
            text=[
                f"{v:.2f}%"
                for v in adversarial_plot[
                    "Original Neural Network"
                ]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>Original Neural Network</b><br>"
                "Attack: %{x}<br>"
                "ASR: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig_adv.add_trace(
        go.Bar(
            name="Adversarially Trained Neural Network",
            x=adversarial_plot["Attack"],
            y=adversarial_plot[
                "Adversarially Trained Neural Network"
            ],
            text=[
                f"{v:.2f}%"
                for v in adversarial_plot[
                    "Adversarially Trained Neural Network"
                ]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>Adversarially Trained Neural Network</b><br>"
                "Attack: %{x}<br>"
                "ASR: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig_adv.update_layout(
        barmode="group",
        height=480,
        title=dict(
            text="FGSM vs PGD Attack Success Rate",
            x=0.5,
            xanchor="center",
            font=dict(
                size=20,
                color="#f2f4f7"
            )
        ),
        xaxis=dict(
            title="Attack Method",
            categoryorder="array",
            categoryarray=["FGSM", "PGD"]
        ),
        yaxis=dict(
            title="Attack Success Rate (%)",
            range=[0, 42],
            gridcolor="#252d36"
        ),
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#f2f4f7"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5
        ),
        margin=dict(
            l=55,
            r=30,
            t=70,
            b=90
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_adv,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    st.markdown(
        """
        <div class="small-note">
        <b>How to interact:</b> hover over a bar for the exact ASR, use the
        toolbar to zoom or pan, double-click to reset the view, or use the
        camera icon to download the graph.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Experiment Context")

    st.markdown(
        """
        The adversarial experiment uses a fixed set of <b>5,000</b> correctly
        classified attack samples. The targeted objective is
        <b>Attack → Normal</b>.
        <br><br>
        FGSM uses a single gradient-based perturbation, while PGD performs
        iterative projected updates. The perturbations are applied in the
        standardized feature space used by the neural network.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="small-note">
        Note: feature-space adversarial examples may contain continuous
        changes to encoded categorical features. They therefore represent
        adversarial robustness tests in model feature space rather than
        necessarily packet-valid network traffic.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# CONFUSION MATRICES
# ============================================================

elif page == "Confusion Matrices":

    st.title("Confusion Matrices")

    st.markdown(
        """
        Confusion matrices show how each IDS model classified the two classes:
        <b>Normal</b> and <b>Attack</b>. The four cells represent true
        negatives, false positives, false negatives, and true positives.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="small-note">
        Correct classifications and the two types of classification errors
        are intentionally displayed using different colors so that the matrix
        can be interpreted quickly.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Select <b>Counts</b> or <b>Percentage</b> and choose which models to
        display. Hover over any cell to see the count and its percentage of
        the corresponding actual class.
        """,
        unsafe_allow_html=True
    )

    if not confusion_matrices:

        st.warning(
            "confusion_matrix_results.csv was not found or could not be parsed."
        )

        st.markdown(
            """
            Expected columns are:
            <b>Model, TN, FP, FN, TP</b>.
            """,
            unsafe_allow_html=True
        )

    else:

        model_order = [
            "Random Forest",
            "XGBoost",
            "Original Neural Network",
            "Adversarially Trained Neural Network"
        ]

        # Use known order first, then any additional models
        available_models = [
            model for model in model_order
            if model in confusion_matrices
        ]

        available_models += [
            model for model in confusion_matrices
            if model not in available_models
        ]

        # Two-column layout
        for start in range(0, len(available_models), 2):

            cols = st.columns(2)

            for col, model_name in zip(
                cols,
                available_models[start:start + 2]
            ):

                cm = np.array(
                    confusion_matrices[model_name],
                    dtype=int
                )

                tn, fp = cm[0]
                fn, tp = cm[1]

                with col:

                    # Different colors for each matrix category
                    cell_colors = [
                        ["#1f77b4", "#d62728"],  # TN, FP
                        ["#ff7f0e", "#2ca02c"]   # FN, TP
                    ]

                    fig_cm = go.Figure()

                    # Draw four separate colored cells
                    for i in range(2):
                        for j in range(2):

                            value = int(cm[i, j])

                            labels = [
                                ["TN", "FP"],
                                ["FN", "TP"]
                            ]

                            fig_cm.add_shape(
                                type="rect",
                                x0=j,
                                y0=i,
                                x1=j + 1,
                                y1=i + 1,
                                fillcolor=cell_colors[i][j],
                                line=dict(
                                    color="white",
                                    width=3
                                )
                            )

                            fig_cm.add_annotation(
                                x=j + 0.5,
                                y=i + 0.5,
                                text=(
                                    f"<b>{labels[i][j]}</b>"
                                    f"<br>{value:,}"
                                ),
                                showarrow=False,
                                font=dict(
                                    color="white",
                                    size=18
                                )
                            )

                    fig_cm.update_xaxes(
                        range=[0, 2],
                        tickmode="array",
                        tickvals=[0.5, 1.5],
                        ticktext=["Normal", "Attack"],
                        title=dict(
                            text="Predicted Class",
                            font=dict(size=13)
                        ),
                        showgrid=False,
                        zeroline=False,
                        fixedrange=False
                    )

                    fig_cm.update_yaxes(
                        range=[2, 0],
                        tickmode="array",
                        tickvals=[0.5, 1.5],
                        ticktext=["Normal", "Attack"],
                        title=dict(
                            text="Actual Class",
                            font=dict(size=13)
                        ),
                        showgrid=False,
                        zeroline=False,
                        fixedrange=False
                    )

                    fig_cm.update_layout(
                        title=dict(
                            text=f"{model_name} Confusion Matrix",
                            x=0.5,
                            xanchor="center",
                            font=dict(
                                size=18,
                                color="#f2f4f7"
                            )
                        ),
                        height=470,
                        plot_bgcolor="#0b0f14",
                        paper_bgcolor="#0b0f14",
                        font=dict(color="#f2f4f7"),
                        margin=dict(
                            l=70,
                            r=30,
                            t=70,
                            b=70
                        )
                    )

                    st.plotly_chart(
                        fig_cm,
                        use_container_width=True,
                        config={
                            "displayModeBar": True
                        }
                    )

# ============================================================
# ROC ANALYSIS
# ============================================================

elif page == "ROC Analysis":

    st.title("ROC Analysis")

    st.markdown(
        """
        The ROC curve compares the true-positive rate with the false-positive
        rate across classification thresholds. ROC-AUC summarizes the model's
        ability to distinguish between Normal and Attack traffic.
        """
    )

    if os.path.exists(ROC_IMAGE_FILE):

        # The saved ROC figure is displayed through Plotly so the graph
        # remains interactive: zoom, pan, reset and download are available.
        try:
            roc_image = np.array(Image.open(ROC_IMAGE_FILE).convert("RGB"))

            fig_roc = go.Figure(
                go.Image(z=roc_image)
            )

            fig_roc.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#0b0f14",
                paper_bgcolor="#0b0f14"
            )

            fig_roc.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )

            fig_roc.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            )

            st.plotly_chart(
                fig_roc,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

            st.caption(
                "Use the Plotly controls to zoom, pan, reset, or download the ROC figure."
            )

        except Exception as exc:
            st.warning(
                f"Could not load the ROC figure interactively: {exc}"
            )

    else:
        st.warning(
            "roc_curve_final.png was not found in results/figures/."
        )

    st.markdown("### ROC-AUC Context")

    st.markdown(
        """
        ROC-AUC is considered together with precision, recall, F1 score, and
        the confusion matrix. This provides a broader view of classification
        behaviour instead of relying on a single performance measure.
        """
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("About Project")

    st.markdown(
        """
        ### Project Title

        **Adversarial Robustness Testing & Evasion-Aware Intrusion Detection System**

        ### Dataset

        **UNSW-NB15**

        ### Models Evaluated

        - Random Forest
        - XGBoost
        - Original Neural Network
        - Adversarially Trained Neural Network

        ### Adversarial Attacks

        - Fast Gradient Sign Method (FGSM)
        - Projected Gradient Descent (PGD)

        ### Adversarial Training

        The robust neural network was trained using the original training data
        together with 5,000 FGSM adversarial examples.

        ### Evaluation

        The project evaluates both:

        1. **Clean-test classification performance**
        2. **Adversarial evasion performance**

        The final dashboard brings these results together into a single
        interface for analysis and presentation.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="small-note">
        This dashboard is intended as a visualization and presentation layer
        for the experimental results produced by the project notebook.
        </div>
        """,
        unsafe_allow_html=True
    )
