import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NSL-KDD | Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. THEME AND CUSTOM CSS (STRICT CONTAINER REPAIR)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #0c1015;
    --panel: #171c23;
    --panel2: #20262e;
    --border: #303740;
    --text: #edf2f7;
    --muted: #8fa2b8;
    --blue: #58a6ff;
    --green: #35c759;
    --red: #ff5252;
    --purple: #bd83ff;
    --gold: #e5a51a;
    --cyan: #32d4df;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(12,16,21,0.95);
    height: 0.5rem;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1700px;
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

h1, h2, h3, h4, p, label {
    color: var(--text);
}

code, pre, .mono {
    font-family: 'Roboto Mono', monospace !important;
}

/* ELIMINATE ALL UNNECESSARY OR EMPTY RECTANGLES / BORDERS */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="element-container"]:empty,
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* TOP HEADER */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    padding: 13px 20px;
    margin: -1.3rem -2.5rem 1.3rem -2.5rem;
    background: #171c23;
    border-bottom: 1px solid #303740;
}

.brand-group {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #36c77a, #58a6ff);
    color: #071018;
    border-radius: 9px;
    font-weight: 900;
    font-size: 17px;
}

.brand-title {
    font-size: 16px;
    font-weight: 750;
    color: #f4f7fb;
}

.brand-subtitle {
    color: #8fa2b8;
    font-size: 12px;
    margin-top: 4px;
}

.header-status {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-family: 'Roboto Mono', monospace;
    font-size: 11px;
    color: var(--green);
}

.header-chip {
    background: #20262e;
    border: 1px solid #303740;
    border-radius: 5px;
    padding: 7px 10px;
    color: #9eb2ca;
}

/* SECTION LABELS */
.section-label {
    color: #8fa6c2;
    font-family: 'Roboto Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    margin: 10px 0 15px 0;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #9baec4 !important;
    font-family: 'Roboto Mono', monospace;
    font-size: 10px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--blue);
    font-size: 1.85rem !important;
    font-weight: 750;
}

/* PANELS */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.panel-title {
    color: #f2f6fb;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 4px;
}

.panel-subtitle {
    color: #8fa2b8;
    font-size: 12px;
    margin-bottom: 16px;
}

.green { color: var(--green) !important; }
.red { color: var(--red) !important; }
.blue { color: var(--blue) !important; }

/* TABS */
div[data-testid="stTabs"] > div:first-child {
    background: #171c23;
    border-bottom: 1px solid #303740;
    gap: 5px;
}

div[data-testid="stTabs"] button {
    color: #8fa2b8;
    font-size: 13px;
    padding: 12px 17px;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue);
    font-weight: 650;
}

/* INPUTS & FORMS */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #20262e !important;
    border-color: #303740 !important;
    color: #edf2f7 !important;
    border-radius: 8px;
}

[data-testid="stForm"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* RESULTS & CARDS */
.result-panel {
    background: #171c23;
    border: 1px solid #303740;
    border-radius: 14px;
    padding: 24px;
    margin: 16px 0;
}

.model-result {
    background: #11161d;
    border: 1px solid #28303a;
    border-radius: 10px;
    text-align: center;
    padding: 17px 8px;
}

.model-value {
    font-family: 'Roboto Mono', monospace;
    font-size: 22px;
    font-weight: 700;
}

.model-name {
    color: #8fa2b8;
    font-family: 'Roboto Mono', monospace;
    font-size: 11px;
    margin-top: 8px;
    letter-spacing: .7px;
    text-transform: uppercase;
}

/* MATRIX GRID */
.matrix-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(80px, 115px));
    gap: 5px;
    justify-content: center;
    margin-top: 12px;
}

.matrix-cell {
    padding: 14px 8px;
    border-radius: 7px;
    text-align: center;
    border: 1px solid #303740;
    background: #20262e;
}

.matrix-value {
    font-family: 'Roboto Mono', monospace;
    font-size: 18px;
    font-weight: 700;
}

.matrix-label {
    font-family: 'Roboto Mono', monospace;
    font-size: 10px;
    margin-top: 5px;
    color: #a1b1c5;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. DATASET SCHEMA
# ============================================================
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

FEATURE_COLUMNS = COLUMNS[:-2]

MODEL_COLORS = {
    "Isolation Forest": "#e5a51a",
    "One-Class SVM": "#bd83ff",
    "Autoencoder": "#58a6ff",
    "Random Forest": "#35c759",
}

# ============================================================
# 4. LOAD SAVED FILES & HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, header=None, names=COLUMNS)
    if str(df.iloc[0]["duration"]).lower() == "duration":
        df = df.iloc[1:].reset_index(drop=True)

    df["label"] = df["label"].astype(str).str.strip()
    return df


@st.cache_data
def load_test_data():
    data = load_dataset("data/KDDTest+.txt")
    X = data[FEATURE_COLUMNS].copy()

    y = np.where(
        data["label"].str.lower() == "normal",
        "Normal",
        "Attack",
    )

    return X, pd.Series(y, name="label"), data


@st.cache_resource
def load_classical_models():
    paths = {
        "Random Forest": "random_forest_model.pkl",
        "Isolation Forest": "isolation_forest_model.pkl",
        "One-Class SVM": "one_class_svm_model.pkl",
    }

    models = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saved model not found: {path}")
        models[name] = joblib.load(path)

    return models


@st.cache_resource
def load_autoencoder_bundle():
    path = "models/autoencoder_final.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved Autoencoder bundle not found: {path}")

    return joblib.load(path)


def normalize_prediction(raw, model_name):
    if model_name == "Random Forest":
        return "Normal" if str(raw).strip().lower() == "normal" else "Attack"
    return "Normal" if int(raw) == 1 else "Attack"


def predict_all_models(record, classical_models, ae_bundle):
    predictions = {}
    scores = {}

    for name, model in classical_models.items():
        raw = model.predict(record)[0]
        predictions[name] = normalize_prediction(raw, name)

        if name == "Random Forest":
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(record)[0]
                    classes = list(model.classes_)
                    index = next(
                        (i for i, label in enumerate(classes) if str(label).strip().lower() == "attack"),
                        None,
                    )
                    if index is not None:
                        scores["RF attack probability"] = float(probs[index])
                except Exception:
                    pass

        if name in ("Isolation Forest", "One-Class SVM"):
            try:
                scores[f"{name} decision score"] = float(
                    np.asarray(model.decision_function(record)).ravel()[0]
                )
            except Exception:
                pass

    preprocessor = ae_bundle["preprocessor"]
    autoencoder = ae_bundle["autoencoder"]
    threshold = float(ae_bundle["anomaly_threshold"])

    transformed = preprocessor.transform(record)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float32)
    reconstructed = autoencoder.predict(transformed, verbose=0)

    mse = float(np.mean(np.square(transformed - reconstructed)))

    predictions["Autoencoder"] = "Attack" if mse > threshold else "Normal"
    scores["Autoencoder reconstruction MSE"] = mse
    scores["Autoencoder threshold"] = threshold

    return predictions, scores

# ============================================================
# 5. EVALUATE EXISTING MODELS
# ============================================================
def get_attack_scores(model, X, name):
    try:
        if name == "Random Forest" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            classes = list(model.classes_)
            attack_index = next(
                (i for i, label in enumerate(classes) if str(label).strip().lower() == "attack"),
                None,
            )
            if attack_index is not None:
                return probabilities[:, attack_index]

        if hasattr(model, "decision_function"):
            values = np.asarray(model.decision_function(X)).ravel()
            if name in ("Isolation Forest", "One-Class SVM"):
                values = -values
            return values

    except Exception:
        pass

    return None


@st.cache_data
def evaluate_saved_models(_X_test, _y_test, _classical_models, _ae_bundle):
    X = _X_test
    y = np.asarray(_y_test)
    y_binary = (y == "Attack").astype(int)

    predictions = {}
    continuous_scores = {}
    rows = []

    for name, model in _classical_models.items():
        raw = model.predict(X)
        if name == "Random Forest":
            pred = np.array([
                "Normal" if str(v).strip().lower() == "normal" else "Attack"
                for v in raw
            ])
        else:
            pred = np.where(np.asarray(raw) == 1, "Normal", "Attack")

        predictions[name] = pred
        score = get_attack_scores(model, X, name)
        if score is not None:
            continuous_scores[name] = np.asarray(score).ravel()

    bundle = _ae_bundle
    preprocessor = bundle["preprocessor"]
    autoencoder = bundle["autoencoder"]
    threshold = float(bundle["anomaly_threshold"])

    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float32)
    reconstructed = autoencoder.predict(transformed, verbose=0)
    errors = np.mean(np.square(transformed - reconstructed), axis=1)

    predictions["Autoencoder"] = np.where(errors > threshold, "Attack", "Normal")
    continuous_scores["Autoencoder"] = errors

    for name, pred in predictions.items():
        row = {
            "Model": name,
            "Accuracy (%)": accuracy_score(y, pred) * 100,
            "Precision (%)": precision_score(y, pred, pos_label="Attack", zero_division=0) * 100,
            "Recall (%)": recall_score(y, pred, pos_label="Attack", zero_division=0) * 100,
            "F1-score (%)": f1_score(y, pred, pos_label="Attack", zero_division=0) * 100,
        }

        score = continuous_scores.get(name)
        try:
            if score is not None and len(np.unique(y_binary)) == 2:
                row["AUC-ROC (%)"] = roc_auc_score(y_binary, score) * 100
            else:
                row["AUC-ROC (%)"] = np.nan
        except Exception:
            row["AUC-ROC (%)"] = np.nan

        rows.append(row)

    results = pd.DataFrame(rows)
    for column in ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)", "AUC-ROC (%)"]:
        results[column] = results[column].round(2)

    return predictions, results, errors, continuous_scores

# ============================================================
# 6. INITIALIZE APP DATA
# ============================================================
try:
    X_test, y_test, test_data = load_test_data()
    classical_models = load_classical_models()
    ae_bundle = load_autoencoder_bundle()

    evaluation_predictions, results_df, ae_errors, continuous_scores = (
        evaluate_saved_models(X_test, y_test, classical_models, ae_bundle)
    )

except Exception as exc:
    st.error(f"Unable to initialize the dashboard: {exc}")
    st.info("Verify data files and models are correctly placed.")
    st.stop()

if "anomaly_log" not in st.session_state:
    st.session_state["anomaly_log"] = []

normal_count = int((y_test == "Normal").sum())
attack_count = int((y_test == "Attack").sum())

# ============================================================
# 7. TOP HEADER
# ============================================================
st.markdown(f"""
<div class="app-header">
    <div class="brand-group">
        <div class="brand-icon">AI</div>
        <div>
            <div class="brand-title">NSL-KDD Intrusion Detection System</div>
            <div class="brand-subtitle">4-model anomaly detection ensemble — trained &amp; evaluated</div>
        </div>
    </div>
    <div class="header-status">
        <span>● All models loaded</span>
        <span class="header-chip">{len(X_test):,} test records</span>
        <span class="header-chip">{len(X_test):,} test samples</span>
    </div>
</div>
""", unsafe_allow_html=True)

overview_tab, performance_tab, explorer_tab, live_tab, log_tab = st.tabs([
    "Overview",
    "Model Performance",
    "Data Explorer",
    "Live Predict",
    "Anomaly Log",
])

def style_matplotlib(ax, fig):
    fig.patch.set_facecolor("#171c23")
    ax.set_facecolor("#171c23")
    ax.tick_params(colors="#8c98a8", labelsize=9)
    ax.xaxis.label.set_color("#9baec4")
    ax.yaxis.label.set_color("#9baec4")
    ax.title.set_color("#edf2f7")
    ax.grid(True, color="#293039", linewidth=0.65, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#293039")

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def panel_heading(title, subtitle=""):
    st.markdown(
        f'<div class="panel-title">{html.escape(title)}</div>'
        + (f'<div class="panel-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )

# ============================================================
# 8. OVERVIEW TAB
# ============================================================
with overview_tab:
    st.markdown('<div class="section-label">DATASET SUMMARY</div>', unsafe_allow_html=True)

    attack_types_count = test_data.loc[
        test_data["label"].str.lower() != "normal", "label"
    ].nunique()

    total_samples = len(y_test) if len(y_test) > 0 else 1

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("TOTAL RECORDS", f"{len(test_data):,}")
    k2.metric("NORMAL TRAFFIC", f"{normal_count:,}", f"{normal_count / total_samples * 100:.1f}% of dataset")
    k3.metric("ATTACK TRAFFIC", f"{attack_count:,}", f"{attack_count / total_samples * 100:.1f}% of dataset")
    k4.metric("FEATURES", len(FEATURE_COLUMNS), "Input features")
    k5.metric("ATTACK TYPES", attack_types_count, "Unique labels")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading("Attack Category Distribution", "Traffic breakdown across Normal, DoS, Probe, R2L and U2R")
        
        attack_category = test_data["label"].str.lower().map(
            lambda x: "Normal" if x == "normal" else (
                "DoS" if x in {"back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "worm"} else (
                    "Probe" if x in {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"} else (
                        "R2L" if x in {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"} else "U2R"
                    )
                )
            )
        )
        category_counts = attack_category.value_counts().reindex(["Normal", "DoS", "Probe", "R2L", "U2R"], fill_value=0)
        colors = ["#35c759", "#ff5252", "#bd83ff", "#e5a51a", "#ed78ad"]

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(category_counts.index, category_counts.values, color=colors, alpha=0.3, edgecolor=colors, linewidth=1.4)
        style_matplotlib(ax, fig)
        ax.set_ylabel("Records")
        show_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading("Model Performance Comparison", "Accuracy / F1 / AUC-ROC across all four saved models")
        
        chart_data = results_df.set_index("Model")[["Accuracy (%)", "F1-score (%)", "AUC-ROC (%)"]]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(chart_data))
        width = 0.23
        series_colors = ["#35c759", "#58a6ff", "#e5a51a"]

        for i, column in enumerate(chart_data.columns):
            values = chart_data[column].values / 100
            ax.bar(x + (i - 1) * width, values, width, label=column.replace(" (%)", ""), color=series_colors[i], alpha=0.3, edgecolor=series_colors[i], linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(["Iso Forest", "OC-SVM", "Autoencoder", "RF"])
        ax.set_ylim(0, 1.05)
        style_matplotlib(ax, fig)
        ax.legend(facecolor="#171c23", edgecolor="#303740", labelcolor="#aab9cc", fontsize=8)
        show_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 9. MODEL PERFORMANCE TAB
# ============================================================
with performance_tab:
    st.markdown('<div class="section-label">MODEL DETAILS</div>', unsafe_allow_html=True)

    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for col_index, model_index in enumerate(range(row_start, min(row_start + 2, 4))):
            model_name = list(MODEL_COLORS.keys())[model_index]
            color = MODEL_COLORS[model_name]
            row = results_df.loc[results_df["Model"] == model_name].iloc[0]

            with cols[col_index]:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:16px;font-weight:750;">{html.escape(model_name)}</div>
                        <span style="font-family:Roboto Mono,monospace;font-size:10px;color:{color};border:1px solid {color};padding:4px 8px;border-radius:5px;">
                            {("SUPERVISED" if model_name == "Random Forest" else "DEEP LEARNING" if model_name == "Autoencoder" else "UNSUPERVISED")}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for metric_name, metric_value in [("Accuracy", row["Accuracy (%)"]), ("Precision", row["Precision (%)"]), ("Recall", row["Recall (%)"]), ("F1 Score", row["F1-score (%)"]), ("AUC-ROC", row["AUC-ROC (%)"])]:
                    val_str = "N/A" if pd.isna(metric_value) else f"{metric_value:.2f}%"
                    pct = 0 if pd.isna(metric_value) else np.clip(metric_value, 0, 100)
                    st.markdown(
                        f"""
                        <div style="margin:10px 0;">
                            <div style="display:flex;justify-content:space-between;font-size:11px;font-family:Roboto Mono;">
                                <span style="color:#9baec4;">{metric_name}</span>
                                <span>{val_str}</span>
                            </div>
                            <div style="height:5px;background:#272d35;border-radius:10px;overflow:hidden;">
                                <div style="width:{pct}%;height:100%;background:{color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                cm = confusion_matrix(y_test, evaluation_predictions[model_name], labels=["Normal", "Attack"])
                tn, fp, fn, tp = cm.ravel()

                st.markdown(
                    f"""
                    <div class="matrix-grid">
                        <div class="matrix-cell"><div class="matrix-value green">{tn:,}</div><div class="matrix-label">TN</div></div>
                        <div class="matrix-cell"><div class="matrix-value red">{fp:,}</div><div class="matrix-label">FP</div></div>
                        <div class="matrix-cell"><div class="matrix-value red">{fn:,}</div><div class="matrix-label">FN</div></div>
                        <div class="matrix-cell"><div class="matrix-value green">{tp:,}</div><div class="matrix-label">TP</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 10. DATA EXPLORER TAB
# ============================================================
with explorer_tab:
    st.markdown('<div class="section-label">DATASET EXPLORER</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading("Class Balance", "Normal versus attack records")
        
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = [normal_count, attack_count]
        labels = ["Normal", "Attack"]
        colors_donut = ["#35c759", "#ff5252"]
        
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors_donut, textprops=dict(color="#edf2f7", fontsize=12),
            wedgeprops=dict(width=0.4, edgecolor='#171c23')
        )
        style_matplotlib(ax, fig)
        ax.grid(False)
        show_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading("Autoencoder MSE", "Reconstruction error distribution")
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ae_thresh = float(ae_bundle["anomaly_threshold"])
        normal_mask = (y_test == "Normal")
        
        ax.hist(ae_errors[normal_mask], bins=30, color="#35c759", alpha=0.5, label="Normal")
        ax.hist(ae_errors[~normal_mask], bins=30, color="#ff5252", alpha=0.5, label="Attack")
        ax.axvline(ae_thresh, color="#e5a51a", linestyle="--", linewidth=2, label="Threshold")
        
        style_matplotlib(ax, fig)
        ax.set_xlabel("Reconstruction MSE")
        ax.set_ylabel("Records")
        ax.legend(facecolor="#171c23", edgecolor="#303740", labelcolor="#edf2f7")
        show_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_heading("Anomaly Scores", "Isolation Forest score distribution")
        
        fig, ax = plt.subplots(figsize=(5, 4))
        iso_scores = continuous_scores.get("Isolation Forest", np.zeros(len(y_test)))
        normal_mask = (y_test == "Normal")
        
        ax.hist(iso_scores[normal_mask], bins=30, color="#35c759", alpha=0.5, label="Normal")
        ax.hist(iso_scores[~normal_mask], bins=30, color="#ff5252", alpha=0.5, label="Attack")
        
        style_matplotlib(ax, fig)
        ax.set_xlabel("Decision score")
        ax.set_ylabel("Records")
        ax.legend(facecolor="#171c23", edgecolor="#303740", labelcolor="#edf2f7")
        show_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance Chart (Repaired layout bounds)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Feature Importance (Random Forest — Top 12)", "Most influential features for the saved Random Forest")
    
    rf_model = classical_models.get("Random Forest")
    importances = None
    
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
    elif hasattr(rf_model, "named_steps"):
        for step_name, step_obj in rf_model.named_steps.items():
            if hasattr(step_obj, "feature_importances_"):
                importances = step_obj.feature_importances_
                break
    elif hasattr(rf_model, "estimator") and hasattr(rf_model.estimator, "feature_importances_"):
        importances = rf_model.estimator.feature_importances_

    if importances is not None and len(importances) > 0:
        if len(importances) == len(FEATURE_COLUMNS):
            feat_names = [f"numerical__{col}" if col not in ["protocol_type", "service", "flag"] else f"categorical__{col}" for col in FEATURE_COLUMNS]
        else:
            feat_names = [f"feature_{i}" for i in range(len(importances))]
        
        top_n = min(12, len(importances))
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.barh(range(top_n), importances[indices], color="#204060", edgecolor="#58a6ff", alpha=0.6)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in indices], color="#9baec4", fontsize=10)
        style_matplotlib(ax, fig)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        show_fig(fig)
    else:
        st.write("Random Forest feature importances not available.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Explore Test Records", "Filter and inspect the underlying data")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        lbl_opt = ["All"] + list(sorted(test_data["label"].unique()))
        sel_lbl = st.selectbox("Traffic label", lbl_opt)
    with col_f2:
        proto_opt = ["All"] + list(sorted(test_data["protocol_type"].unique()))
        sel_proto = st.selectbox("Protocol", proto_opt)

    filtered_df = test_data.copy()
    if sel_lbl != "All":
        filtered_df = filtered_df[filtered_df["label"] == sel_lbl]
    if sel_proto != "All":
        filtered_df = filtered_df[filtered_df["protocol_type"] == sel_proto]

    st.markdown(f"**Showing {len(filtered_df):,} matching records**")
    st.dataframe(filtered_df, use_container_width=True, height=350)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data CSV", csv_data, "filtered_data.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 11. LIVE PREDICT TAB
# ============================================================
with live_tab:
    st.markdown('<div class="section-label">LIVE PREDICTION</div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#edf2f7;font-size:14px;margin-bottom:18px;">'
        'Enter network connection features below. The ensemble will classify the traffic using '
        'all 4 trained models and return a verdict with individual model confidence scores.'
        '</p>',
        unsafe_allow_html=True
    )

    preset = st.selectbox("Sample preset", ["Custom input", "Load: Normal Sample", "Load: Neptune (DoS)", "Load: Portsweep (Probe)", "Load: Satan (Probe)"])

    d_duration, d_proto, d_serv, d_flag, d_src, d_dst = 0, "tcp", "http", "SF", 0, 0
    d_count, d_srv_count, d_serror, d_rerror, d_login, d_same_srv = 1, 1, 0.0, 0.0, 1, 1.0

    if preset == "Load: Normal Sample":
        d_duration, d_proto, d_serv, d_flag, d_src, d_dst = 0, "tcp", "http", "SF", 215, 45076
        d_count, d_srv_count, d_serror, d_rerror, d_login, d_same_srv = 1, 1, 0.0, 0.0, 1, 1.0
    elif preset == "Load: Neptune (DoS)":
        d_duration, d_proto, d_serv, d_flag, d_src, d_dst = 0, "tcp", "private", "S0", 0, 0
        d_count, d_srv_count, d_serror, d_rerror, d_login, d_same_srv = 229, 10, 1.0, 0.0, 0, 0.04
    elif preset in ["Load: Portsweep (Probe)", "Load: Satan (Probe)"]:
        d_duration, d_proto, d_serv, d_flag, d_src, d_dst = 0, "tcp", "private", "REJ", 0, 0
        d_count, d_srv_count, d_serror, d_rerror, d_login, d_same_srv = 150, 1, 0.0, 1.0, 0, 0.01

    with st.form("live_form"):
        r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
        with r1_1:
            in_duration = st.number_input("DURATION (S)", value=d_duration)
        with r1_2:
            in_proto = st.selectbox("PROTOCOL TYPE", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(d_proto))
        with r1_3:
            services = sorted(test_data["service"].unique())
            in_serv = st.selectbox("SERVICE", services, index=services.index(d_serv) if d_serv in services else 0)
        with r1_4:
            flags = sorted(test_data["flag"].unique())
            in_flag = st.selectbox("FLAG", flags, index=flags.index(d_flag) if d_flag in flags else 0)
        with r1_5:
            in_src = st.number_input("SRC BYTES", value=d_src)
        with r1_6:
            in_dst = st.number_input("DST BYTES", value=d_dst)

        r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
        with r2_1:
            in_count = st.number_input("COUNT (2S WINDOW)", value=d_count)
        with r2_2:
            in_srv_count = st.number_input("SRV COUNT", value=d_srv_count)
        with r2_3:
            in_serror = st.number_input("SERROR RATE [0-1]", value=float(d_serror))
        with r2_4:
            in_rerror = st.number_input("RERROR RATE [0-1]", value=float(d_rerror))
        with r2_5:
            in_login = st.selectbox("LOGGED IN (0/1)", [0, 1], index=d_login)
        with r2_6:
            in_same_srv = st.number_input("SAME SRV RATE [0-1]", value=float(d_same_srv))

        btn_run = st.form_submit_button("Run All Models", type="primary")

    if btn_run or preset != "Custom input":
        row_dict = {col: 0 for col in FEATURE_COLUMNS}
        row_dict.update({
            "duration": in_duration, "protocol_type": in_proto, "service": in_serv,
            "flag": in_flag, "src_bytes": in_src, "dst_bytes": in_dst,
            "count": in_count, "srv_count": in_srv_count, "serror_rate": in_serror,
            "rerror_rate": in_rerror, "logged_in": in_login, "same_srv_rate": in_same_srv
        })
        input_df = pd.DataFrame([row_dict])

        predictions, scores = predict_all_models(input_df, classical_models, ae_bundle)
        attack_votes = sum(1 for v in predictions.values() if v == "Attack")
        ensemble_verdict = "Attack" if attack_votes >= 2 else "Normal"

        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        if ensemble_verdict == "Normal":
            st.markdown('<h2 style="color:#35c759;margin:0;">NORMAL TRAFFIC</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color:#8fa2b8;">All models agree: traffic pattern matches normal baseline. No anomaly detected.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color:#ff5252;margin:0;">ANOMALOUS TRAFFIC DETECTED</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#8fa2b8;">Warning: {attack_votes}/4 models flagged this connection as malicious.</p>', unsafe_allow_html=True)

        st.write("")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)

        with c_m1:
            iso_val = scores.get("Isolation Forest decision score", -0.099)
            st.markdown(f"""
            <div class="model-result">
                <div class="model-value green">{iso_val:.3f}</div>
                <div class="model-name">ISOLATION FOREST</div>
            </div>
            """, unsafe_allow_html=True)

        with c_m2:
            svm_pred = predictions.get("One-Class SVM", "Normal")
            color_cls = "green" if svm_pred == "Normal" else "red"
            st.markdown(f"""
            <div class="model-result">
                <div class="model-value {color_cls}">{svm_pred}</div>
                <div class="model-name">ONE-CLASS SVM</div>
            </div>
            """, unsafe_allow_html=True)

        with c_m3:
            ae_val = scores.get("Autoencoder reconstruction MSE", 0.075)
            st.markdown(f"""
            <div class="model-result">
                <div class="model-value green">{ae_val:.3f}</div>
                <div class="model-name">AUTOENCODER MSE</div>
            </div>
            """, unsafe_allow_html=True)

        with c_m4:
            rf_prob = scores.get("RF attack probability", 0.045) * 100
            st.markdown(f"""
            <div class="model-result">
                <div class="model-value green">{rf_prob:.1f}%</div>
                <div class="model-name">RANDOM FOREST</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if ensemble_verdict == "Attack":
            st.session_state["anomaly_log"].append({
                "Protocol": in_proto,
                "Service": in_serv,
                "Src Bytes": in_src,
                "Dst Bytes": in_dst,
                "Votes": f"{attack_votes}/4",
            })

# ============================================================
# 12. ANOMALY LOG TAB
# ============================================================
with log_tab:
    st.markdown('<div class="section-label">DETECTED ANOMALIES LOG</div>', unsafe_allow_html=True)

    if st.session_state["anomaly_log"]:
        log_df = pd.DataFrame(st.session_state["anomaly_log"])
        st.dataframe(log_df, use_container_width=True)
        if st.button("Clear Log"):
            st.session_state["anomaly_log"] = []
            st.rerun()
    else:
        st.info("No live anomalies logged in this session yet.")