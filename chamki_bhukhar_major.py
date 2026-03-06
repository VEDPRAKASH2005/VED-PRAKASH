"""
╔══════════════════════════════════════════════════════════════════════╗
║     CHAMKI BHUKHAR PREDICTION SYSTEM - MAJOR PROJECT               ║
║     Advanced ML System with Flask Web App                           ║
║     Features: Multi-Model Comparison, Feature Importance,           ║
║               Patient History, Real-time Prediction                 ║
║     Author  : Ved Prakash Kumar | Govt. Polytechnic Motihari        ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python3 chamki_bhukhar_major.py

Then open browser and go to: http://localhost:5000
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os, json, csv, warnings, io, base64
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                       f1_score, roc_auc_score, confusion_matrix,
                                       roc_curve, classification_report)
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.pipeline          import Pipeline

from flask import Flask, request, jsonify, render_template_string, redirect, url_for

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "chamki_dataset.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "saved_models")
HISTORY_PATH  = os.path.join(BASE_DIR, "patient_history.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATASET GENERATION  (1200 realistic rows)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "age", "gender", "nutrition_status", "region",
    "season", "temperature", "humidity", "rainfall",
    "fever_days", "vomiting", "seizure", "unconscious",
    "lychee_consumed", "blood_sugar", "body_temp"
]
LABEL_COL = "risk_label"

def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        age            = int(rng.integers(1, 15))
        gender         = int(rng.integers(0, 2))           # 0=F, 1=M
        nutrition      = int(rng.integers(0, 3))           # 0=good,1=moderate,2=poor
        region         = int(rng.integers(0, 4))           # 0-3 = districts
        season         = int(rng.integers(0, 4))           # 0=winter…3=summer
        temperature    = round(float(rng.uniform(25, 48)), 1)
        humidity       = round(float(rng.uniform(40, 100)), 1)
        rainfall       = round(float(rng.uniform(0, 250)), 1)
        fever_days     = int(rng.integers(0, 8))
        vomiting       = int(rng.integers(0, 2))
        seizure        = int(rng.integers(0, 2))
        unconscious    = int(rng.integers(0, 2))
        lychee         = int(rng.integers(0, 2))
        blood_sugar    = round(float(rng.uniform(40, 120)), 1)
        body_temp      = round(float(rng.uniform(36.0, 42.5)), 1)

        # Risk scoring (domain-based heuristic)
        score = 0
        score += 3 if age < 6                   else (1 if age < 10 else 0)
        score += 2 if nutrition == 2             else (1 if nutrition == 1 else 0)
        score += 2 if season == 3                else 0           # summer
        score += 2 if temperature > 40           else (1 if temperature > 37 else 0)
        score += 1 if humidity > 75              else 0
        score += 2 if fever_days >= 2            else 0
        score += 3 if seizure == 1               else 0
        score += 3 if unconscious == 1           else 0
        score += 2 if lychee == 1                else 0
        score += 2 if blood_sugar < 60           else 0
        score += 2 if body_temp > 39.5           else (1 if body_temp > 38.5 else 0)
        score += 1 if vomiting == 1              else 0

        # Add noise
        score += int(rng.integers(-1, 3))
        label = 1 if score >= 10 else 0           # 1=High Risk, 0=Low Risk

        rows.append([age, gender, nutrition, region, season,
                     temperature, humidity, rainfall,
                     fever_days, vomiting, seizure, unconscious,
                     lychee, blood_sugar, body_temp, label])

    df = pd.DataFrame(rows, columns=FEATURE_COLS + [LABEL_COL])
    df.to_csv(DATA_PATH, index=False)
    print(f"[✓] Dataset generated: {len(df)} rows → {DATA_PATH}")
    return df


# ─────────────────────────────────────────────
# 2. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=42),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        random_state=42),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=42))]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(probability=True, class_weight="balanced", random_state=42))])
}

results_cache = {}          # model_name → metrics dict
best_model_name = None
best_model_obj  = None


def train_all_models(df):
    global best_model_name, best_model_obj, results_cache
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_cache = {}
    best_auc = 0

    print("\n[⚙] Training all models...\n")
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_acc = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="accuracy", n_jobs=-1).mean()
        metrics = {
            "accuracy" : round(accuracy_score (y_test, y_pred)  * 100, 2),
            "precision": round(precision_score(y_test, y_pred)  * 100, 2),
            "recall"   : round(recall_score   (y_test, y_pred)  * 100, 2),
            "f1"       : round(f1_score       (y_test, y_pred)  * 100, 2),
            "roc_auc"  : round(roc_auc_score  (y_test, y_proba) * 100, 2),
            "cv_acc"   : round(cv_acc          * 100, 2),
            "cm"       : confusion_matrix(y_test, y_pred).tolist(),
            "fpr"      : roc_curve(y_test, y_proba)[0].tolist(),
            "tpr"      : roc_curve(y_test, y_proba)[1].tolist(),
        }
        results_cache[name] = metrics
        print(f"  {name:25s} | Acc={metrics['accuracy']}% | AUC={metrics['roc_auc']}%")

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.replace(' ','_')}.pkl"))

        if metrics["roc_auc"] > best_auc:
            best_auc        = metrics["roc_auc"]
            best_model_name = name
            best_model_obj  = model

    # Feature importance (Random Forest)
    rf = MODELS["Random Forest"]
    fi = dict(zip(FEATURE_COLS, rf.feature_importances_))
    joblib.dump(fi, os.path.join(MODEL_DIR, "feature_importance.pkl"))
    print(f"\n[★] Best model: {best_model_name} (AUC={best_auc}%)")
    return results_cache


# ─────────────────────────────────────────────
# 3. CHART HELPERS  (return base64 PNG strings)
# ─────────────────────────────────────────────
COLORS = ["#0b5394", "#1a9641", "#d7191c", "#fdae61", "#7b2d8b"]

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def chart_model_comparison():
    names   = list(results_cache.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, l) in enumerate(zip(metrics, labels)):
        vals = [results_cache[n][m] for n in names]
        ax.bar(x + i*w, vals, w, label=l, color=COLORS[i], alpha=0.85)
    ax.set_xticks(x + 2*w)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(50, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_roc_curves():
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, m) in enumerate(results_cache.items()):
        ax.plot(m["fpr"], m["tpr"], color=COLORS[i],
                label=f"{name} (AUC={m['roc_auc']}%)", linewidth=2)
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def chart_feature_importance():
    fi = joblib.load(os.path.join(MODEL_DIR, "feature_importance.pkl"))
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    readable = {
        "age":"Age", "gender":"Gender", "nutrition_status":"Nutrition",
        "region":"Region", "season":"Season", "temperature":"Env. Temp",
        "humidity":"Humidity", "rainfall":"Rainfall",
        "fever_days":"Fever Days", "vomiting":"Vomiting",
        "seizure":"Seizure", "unconscious":"Unconscious",
        "lychee_consumed":"Lychee", "blood_sugar":"Blood Sugar",
        "body_temp":"Body Temp"
    }
    names  = [readable.get(k, k) for k in sorted_fi]
    values = list(sorted_fi.values())
    colors = ["#d7191c" if v > 0.10 else "#0b5394" if v > 0.05 else "#74add1"
              for v in values]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Random Forest — SHAP Style)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_confusion_matrix(model_name):
    cm = np.array(results_cache[model_name]["cm"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low Risk","High Risk"],
                yticklabels=["Low Risk","High Risk"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_risk_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df[LABEL_COL].value_counts()
    axes[0].pie(counts, labels=["Low Risk","High Risk"],
                colors=["#1a9641","#d7191c"], autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white",linewidth=2))
    axes[0].set_title("Risk Distribution", fontweight="bold")
    age_groups = pd.cut(df["age"], bins=[0,4,9,14], labels=["1-4","5-9","10-14"])
    grp = df.groupby([age_groups, LABEL_COL]).size().unstack(fill_value=0)
    grp.plot(kind="bar", ax=axes[1], color=["#1a9641","#d7191c"],
             edgecolor="white", legend=True)
    axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Count")
    axes[1].set_title("Risk by Age Group", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────
# 4. PATIENT HISTORY
# ─────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:100]                    # keep latest 100
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────
# 5. FLASK WEB APPLICATION
# ─────────────────────────────────────────────
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chamki Bhukhar Prediction System</title>
<style>
  :root{--blue:#0b5394;--red:#d7191c;--green:#1a9641;--light:#f0f4ff;--card:#fff;}
  *{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',sans-serif;}
  body{background:var(--light);color:#222;}

  /* NAV */
  nav{background:var(--blue);color:#fff;padding:14px 24px;display:flex;
      align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;
      box-shadow:0 3px 12px #0004;}
  nav .logo{font-size:1.25rem;font-weight:700;}
  nav .sub{font-size:.78rem;opacity:.8;}
  nav .tabs{display:flex;gap:6px;flex-wrap:wrap;}
  nav .tabs a{color:#fff;text-decoration:none;padding:7px 14px;border-radius:6px;
              font-size:.88rem;transition:.2s;}
  nav .tabs a:hover,nav .tabs a.active{background:#fff3;font-weight:600;}

  /* MAIN */
  .main{max-width:1100px;margin:30px auto;padding:0 16px;}

  /* SECTION */
  .section{display:none;} .section.active{display:block;}

  /* CARD */
  .card{background:var(--card);border-radius:14px;padding:24px 28px;
        box-shadow:0 2px 14px #0001;margin-bottom:22px;}
  .card h2{font-size:1.15rem;color:var(--blue);margin-bottom:16px;
           display:flex;align-items:center;gap:8px;}

  /* FORM GRID */
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px;}
  label{font-size:.82rem;font-weight:600;color:#444;display:block;margin-bottom:4px;}
  input,select{width:100%;padding:9px 12px;border:1.5px solid #ccd6f0;
               border-radius:8px;font-size:.9rem;transition:.2s;outline:none;}
  input:focus,select:focus{border-color:var(--blue);}
  .btn{background:var(--blue);color:#fff;border:none;padding:13px 32px;
       border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;
       transition:.2s;letter-spacing:.5px;}
  .btn:hover{background:#073e6e;transform:translateY(-1px);}
  .btn-red{background:var(--red);}
  .btn-red:hover{background:#a01010;}

  /* RESULT */
  .result-box{padding:22px 28px;border-radius:12px;margin-top:18px;
              text-align:center;display:none;}
  .result-box.high{background:#fdecea;border:2px solid var(--red);}
  .result-box.low {background:#edfbef;border:2px solid var(--green);}
  .result-box h3{font-size:1.6rem;margin-bottom:8px;}
  .result-box p {font-size:.95rem;color:#555;}
  .prob-bar{height:18px;border-radius:10px;background:#e0e0e0;
            margin:12px 0;overflow:hidden;}
  .prob-fill{height:100%;border-radius:10px;transition:width 1s ease;}

  /* METRICS TABLE */
  table{width:100%;border-collapse:collapse;font-size:.9rem;}
  th{background:var(--blue);color:#fff;padding:10px 14px;text-align:left;}
  td{padding:9px 14px;border-bottom:1px solid #e8ecf5;}
  tr:hover td{background:#f5f7ff;}
  .badge{display:inline-block;padding:3px 10px;border-radius:20px;
         font-size:.78rem;font-weight:700;}
  .badge.best{background:#fff4cc;color:#7a5c00;}

  /* CHARTS */
  .chart-img{width:100%;border-radius:10px;margin-top:10px;}
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
  @media(max-width:650px){.chart-grid{grid-template-columns:1fr;}}

  /* HISTORY */
  .hist-item{background:#f8faff;border-radius:10px;padding:14px 18px;
             margin-bottom:10px;display:flex;justify-content:space-between;
             align-items:flex-start;flex-wrap:wrap;gap:8px;
             border-left:4px solid var(--blue);}
  .hist-item.high-risk{border-left-color:var(--red);}
  .hist-tag{padding:3px 10px;border-radius:20px;font-size:.78rem;font-weight:700;}
  .tag-high{background:#fdecea;color:var(--red);}
  .tag-low {background:#edfbef;color:var(--green);}

  /* ABOUT */
  .about-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  @media(max-width:600px){.about-grid{grid-template-columns:1fr;}}
  .info-item{background:#f5f7ff;border-radius:10px;padding:14px 16px;
             border-left:4px solid var(--blue);}
  .info-item h4{color:var(--blue);margin-bottom:6px;}
  .info-item p{font-size:.88rem;color:#555;line-height:1.6;}

  /* LOADER */
  .loader{display:none;text-align:center;padding:20px;color:var(--blue);font-weight:600;}
  .spinner{width:36px;height:36px;border:4px solid #ccd6f0;
           border-top-color:var(--blue);border-radius:50%;
           animation:spin .8s linear infinite;margin:0 auto 10px;}
  @keyframes spin{to{transform:rotate(360deg);}}

  .section-title{font-size:1.5rem;font-weight:700;color:var(--blue);
                 margin-bottom:4px;}
  .section-sub{color:#666;font-size:.9rem;margin-bottom:20px;}
  .divider{height:1px;background:#e0e8f5;margin:20px 0;}
  .model-select-row{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap;}
</style>
</head>
<body>

<nav>
  <div>
    <div class="logo">🧠 Chamki Bhukhar Prediction System</div>
    <div class="sub">Major Project — Govt. Polytechnic Motihari | CSE Dept.</div>
  </div>
  <div class="tabs">
    <a href="#" class="active" onclick="showTab('predict',this)">🔍 Predict</a>
    <a href="#" onclick="showTab('analysis',this)">📊 Analysis</a>
    <a href="#" onclick="showTab('history',this)">📋 History</a>
    <a href="#" onclick="showTab('about',this)">ℹ About</a>
  </div>
</nav>

<div class="main">

<!-- ===== PREDICT ===== -->
<div id="tab-predict" class="section active">
  <div class="section-title">AES Risk Prediction</div>
  <div class="section-sub">Patient ka data fill karo — model turant High/Low Risk batayega</div>

  <div class="card">
    <h2>👤 Patient Details</h2>
    <div class="grid">
      <div>
        <label>Patient Name</label>
        <input id="p_name" type="text" placeholder="e.g. Rahul Kumar"/>
      </div>
      <div>
        <label>Age (years)</label>
        <input id="age" type="number" min="1" max="14" value="5"/>
      </div>
      <div>
        <label>Gender</label>
        <select id="gender">
          <option value="0">Female</option>
          <option value="1">Male</option>
        </select>
      </div>
      <div>
        <label>Nutrition Status</label>
        <select id="nutrition_status">
          <option value="0">Good</option>
          <option value="1">Moderate</option>
          <option value="2">Poor (Malnourished)</option>
        </select>
      </div>
      <div>
        <label>Region / District</label>
        <select id="region">
          <option value="0">Motihari</option>
          <option value="1">Muzaffarpur</option>
          <option value="2">Vaishali</option>
          <option value="3">Sitamarhi</option>
        </select>
      </div>
      <div>
        <label>Season</label>
        <select id="season">
          <option value="0">Winter</option>
          <option value="1">Spring</option>
          <option value="2">Monsoon</option>
          <option value="3">Summer (High Risk)</option>
        </select>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🌡 Environmental & Clinical Data</h2>
    <div class="grid">
      <div>
        <label>Env. Temperature (°C)</label>
        <input id="temperature" type="number" step="0.1" min="25" max="50" value="37.0"/>
      </div>
      <div>
        <label>Humidity (%)</label>
        <input id="humidity" type="number" step="0.1" min="30" max="100" value="70"/>
      </div>
      <div>
        <label>Rainfall (mm)</label>
        <input id="rainfall" type="number" step="0.1" min="0" max="300" value="50"/>
      </div>
      <div>
        <label>Fever Duration (days)</label>
        <input id="fever_days" type="number" min="0" max="10" value="2"/>
      </div>
      <div>
        <label>Body Temperature (°C)</label>
        <input id="body_temp" type="number" step="0.1" min="36" max="43" value="38.5"/>
      </div>
      <div>
        <label>Blood Sugar (mg/dL)</label>
        <input id="blood_sugar" type="number" step="0.1" min="30" max="150" value="80"/>
      </div>
    </div>

    <div class="divider"></div>
    <h2>⚠ Symptoms</h2>
    <div class="grid">
      <div>
        <label>Vomiting</label>
        <select id="vomiting">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>Seizure / Daura</label>
        <select id="seizure">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>Unconscious / Lethargy</label>
        <select id="unconscious">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>Lychee Consumed</label>
        <select id="lychee_consumed">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
    </div>

    <div class="divider"></div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <div>
        <label>Select ML Model</label>
        <select id="model_select">
          <option>Random Forest</option>
          <option>Gradient Boosting</option>
          <option>Logistic Regression</option>
          <option>SVM</option>
        </select>
      </div>
      <div style="margin-top:22px;">
        <button class="btn" onclick="predict()">🔍 Predict Risk</button>
      </div>
    </div>

    <div class="loader" id="loader">
      <div class="spinner"></div>Analyzing patient data...
    </div>

    <div class="result-box" id="result-box">
      <h3 id="result-title"></h3>
      <div class="prob-bar"><div class="prob-fill" id="prob-fill"></div></div>
      <p id="result-prob"></p>
      <p id="result-advice" style="margin-top:10px;font-weight:600;"></p>
    </div>
  </div>
</div>

<!-- ===== ANALYSIS ===== -->
<div id="tab-analysis" class="section">
  <div class="section-title">📊 Model Analysis Dashboard</div>
  <div class="section-sub">Sabhi ML models ka performance comparison aur feature importance</div>

  <div class="card">
    <h2>🏆 Model Performance Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>Precision</th>
          <th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>CV Acc.</th>
        </tr>
      </thead>
      <tbody id="metrics-table"></tbody>
    </table>
    <img id="chart-comparison" class="chart-img" style="margin-top:18px;" src=""/>
  </div>

  <div class="chart-grid">
    <div class="card">
      <h2>📈 ROC Curves</h2>
      <img id="chart-roc" class="chart-img" src=""/>
    </div>
    <div class="card">
      <h2>🌟 Feature Importance (SHAP Style)</h2>
      <img id="chart-fi" class="chart-img" src=""/>
    </div>
  </div>

  <div class="card">
    <h2>🔲 Confusion Matrix</h2>
    <div class="model-select-row">
      <div>
        <label>Select Model:</label>
        <select id="cm-model-select" onchange="loadCM()">
          <option>Random Forest</option>
          <option>Gradient Boosting</option>
          <option>Logistic Regression</option>
          <option>SVM</option>
        </select>
      </div>
    </div>
    <img id="chart-cm" class="chart-img" style="max-width:420px;" src=""/>
  </div>

  <div class="card">
    <h2>📊 Dataset Risk Distribution</h2>
    <img id="chart-dist" class="chart-img" src=""/>
  </div>
</div>

<!-- ===== HISTORY ===== -->
<div id="tab-history" class="section">
  <div class="section-title">📋 Patient Prediction History</div>
  <div class="section-sub">Last 100 predictions ka record — local JSON mein save hota hai</div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <span id="hist-count" style="font-weight:600;color:var(--blue);"></span>
      <button class="btn btn-red" onclick="clearHistory()" style="padding:8px 18px;font-size:.85rem;">
        🗑 Clear All
      </button>
    </div>
    <div id="history-list"></div>
  </div>
</div>

<!-- ===== ABOUT ===== -->
<div id="tab-about" class="section">
  <div class="section-title">ℹ About This Project</div>
  <div class="section-sub">Chamki Bhukhar (AES) Major Project — Advanced ML System</div>

  <div class="card">
    <h2>🎓 Project Info</h2>
    <div class="about-grid">
      <div class="info-item">
        <h4>Project Name</h4>
        <p>Chamki Bhukhar Prediction System (Major Project)</p>
      </div>
      <div class="info-item">
        <h4>Student</h4>
        <p>Ved Prakash Kumar | Roll No: 23-CSE-31</p>
      </div>
      <div class="info-item">
        <h4>Institute</h4>
        <p>Govt. Polytechnic Motihari, East Champaran, Bihar</p>
      </div>
      <div class="info-item">
        <h4>Guide</h4>
        <p>Prof. Satya Deo Kumar Ram & Prof. Rajan Kumar</p>
      </div>
      <div class="info-item">
        <h4>Disease</h4>
        <p>Acute Encephalitis Syndrome (AES) — mostly affects children under 15 in Bihar</p>
      </div>
      <div class="info-item">
        <h4>Dataset</h4>
        <p>1200 rows synthetic + domain-realistic data with 15 features</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🤖 ML Models Used</h2>
    <div class="about-grid">
      <div class="info-item">
        <h4>🌲 Random Forest</h4>
        <p>Ensemble of decision trees. Best for mixed numerical/categorical data. Robust against overfitting.</p>
      </div>
      <div class="info-item">
        <h4>📈 Gradient Boosting</h4>
        <p>Sequential boosting algorithm. Learns from previous errors. Often highest accuracy on tabular data.</p>
      </div>
      <div class="info-item">
        <h4>📐 Logistic Regression</h4>
        <p>Statistical baseline model. Fast, interpretable, good for linearly separable data.</p>
      </div>
      <div class="info-item">
        <h4>⚡ SVM</h4>
        <p>Support Vector Machine. Finds optimal hyperplane. Good on small-medium datasets.</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>✨ Major Upgrades from Minor Project</h2>
    <div class="about-grid">
      <div class="info-item"><h4>📊 Dataset</h4><p>10 rows → 1200 rows realistic synthetic data</p></div>
      <div class="info-item"><h4>🤖 Models</h4><p>1 model → 4 models with comparison</p></div>
      <div class="info-item"><h4>🌐 Interface</h4><p>Tkinter GUI → Full Flask Web App</p></div>
      <div class="info-item"><h4>🌟 Explainability</h4><p>None → Feature Importance (SHAP Style)</p></div>
      <div class="info-item"><h4>📋 History</h4><p>None → Patient history save system</p></div>
      <div class="info-item"><h4>📈 Evaluation</h4><p>Basic accuracy → ROC, CV, Confusion Matrix</p></div>
    </div>
  </div>
</div>

</div><!-- end .main -->

<script>
// ── TAB NAVIGATION ──────────────────────────
function showTab(tab, el){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('nav .tabs a').forEach(a=>a.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  el.classList.add('active');
  if(tab==='analysis') loadAnalysis();
  if(tab==='history')  loadHistory();
}

// ── PREDICT ─────────────────────────────────
async function predict(){
  const name = document.getElementById('p_name').value || 'Patient';
  const payload = {
    name,
    model: document.getElementById('model_select').value,
    features: {
      age:             +document.getElementById('age').value,
      gender:          +document.getElementById('gender').value,
      nutrition_status:+document.getElementById('nutrition_status').value,
      region:          +document.getElementById('region').value,
      season:          +document.getElementById('season').value,
      temperature:     +document.getElementById('temperature').value,
      humidity:        +document.getElementById('humidity').value,
      rainfall:        +document.getElementById('rainfall').value,
      fever_days:      +document.getElementById('fever_days').value,
      vomiting:        +document.getElementById('vomiting').value,
      seizure:         +document.getElementById('seizure').value,
      unconscious:     +document.getElementById('unconscious').value,
      lychee_consumed: +document.getElementById('lychee_consumed').value,
      blood_sugar:     +document.getElementById('blood_sugar').value,
      body_temp:       +document.getElementById('body_temp').value,
    }
  };

  document.getElementById('loader').style.display='block';
  document.getElementById('result-box').style.display='none';

  const resp = await fetch('/api/predict', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  document.getElementById('loader').style.display='none';

  const box = document.getElementById('result-box');
  const pct = (data.probability*100).toFixed(1);
  box.className = 'result-box ' + (data.risk==='High Risk' ? 'high' : 'low');
  document.getElementById('result-title').textContent =
    (data.risk==='High Risk' ? '⚠️ HIGH RISK' : '✅ LOW RISK') + ' — ' + name;
  document.getElementById('result-prob').textContent =
    `Risk Probability: ${pct}%  |  Model: ${data.model}  |  Confidence: ${data.confidence}`;
  document.getElementById('result-advice').textContent = data.advice;
  const fill = document.getElementById('prob-fill');
  fill.style.width = '0%';
  fill.style.background = data.risk==='High Risk' ? '#d7191c' : '#1a9641';
  setTimeout(()=>{ fill.style.width = pct+'%'; }, 100);
  box.style.display = 'block';
}

// ── ANALYSIS ────────────────────────────────
let analysisLoaded = false;
async function loadAnalysis(){
  if(analysisLoaded) return;
  const r = await fetch('/api/analysis');
  const d = await r.json();

  // Metrics table
  const tbody = document.getElementById('metrics-table');
  tbody.innerHTML = '';
  for(const [name, m] of Object.entries(d.metrics)){
    const isBest = name===d.best_model;
    tbody.innerHTML += `<tr>
      <td>${name} ${isBest?'<span class="badge best">★ Best</span>':''}</td>
      <td>${m.accuracy}%</td><td>${m.precision}%</td>
      <td>${m.recall}%</td><td>${m.f1}%</td>
      <td>${m.roc_auc}%</td><td>${m.cv_acc}%</td>
    </tr>`;
  }

  document.getElementById('chart-comparison').src = 'data:image/png;base64,'+d.chart_comparison;
  document.getElementById('chart-roc').src        = 'data:image/png;base64,'+d.chart_roc;
  document.getElementById('chart-fi').src         = 'data:image/png;base64,'+d.chart_fi;
  document.getElementById('chart-dist').src       = 'data:image/png;base64,'+d.chart_dist;
  loadCM();
  analysisLoaded = true;
}

async function loadCM(){
  const model = document.getElementById('cm-model-select').value;
  const r = await fetch('/api/cm?model='+encodeURIComponent(model));
  const d = await r.json();
  document.getElementById('chart-cm').src = 'data:image/png;base64,'+d.chart;
}

// ── HISTORY ─────────────────────────────────
async function loadHistory(){
  const r = await fetch('/api/history');
  const d = await r.json();
  const list = document.getElementById('history-list');
  document.getElementById('hist-count').textContent = `Total Predictions: ${d.length}`;
  if(d.length===0){
    list.innerHTML='<p style="color:#888;text-align:center;padding:20px;">Koi prediction abhi nahi hai.</p>';
    return;
  }
  list.innerHTML = d.map(h=>`
    <div class="hist-item ${h.risk==='High Risk'?'high-risk':''}">
      <div>
        <strong>${h.name}</strong>
        <span style="color:#888;font-size:.8rem;margin-left:10px;">${h.time}</span><br/>
        <span style="font-size:.85rem;color:#555;">
          Age: ${h.age} | Model: ${h.model} | Prob: ${(h.probability*100).toFixed(1)}%
        </span>
      </div>
      <span class="hist-tag ${h.risk==='High Risk'?'tag-high':'tag-low'}">${h.risk}</span>
    </div>
  `).join('');
}

async function clearHistory(){
  if(!confirm('Saari history delete karein?')) return;
  await fetch('/api/history/clear', {method:'POST'});
  loadHistory();
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
FEATURE_ORDER = FEATURE_COLS   # global order

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    feat = data["features"]
    name  = data.get("name", "Patient")
    mname = data.get("model", "Random Forest")

    # Load model
    mpath = os.path.join(MODEL_DIR, f"{mname.replace(' ','_')}.pkl")
    model = joblib.load(mpath)

    X = np.array([[feat[f] for f in FEATURE_ORDER]])
    prob = float(model.predict_proba(X)[0][1])
    risk = "High Risk" if prob > 0.5 else "Low Risk"

    conf_val = abs(prob - 0.5) * 2        # 0→1
    if   conf_val > 0.7: conf = "Very High"
    elif conf_val > 0.4: conf = "High"
    elif conf_val > 0.2: conf = "Moderate"
    else:                conf = "Low"

    advice = ("⚠ Turant doctor se mile! Nearest hospital mein le jaayein."
              if risk == "High Risk"
              else "✅ Low risk hai. Hydration maintain karein aur observe karte rahein.")

    record = {
        "name": name, "risk": risk, "probability": round(prob, 4),
        "model": mname, "confidence": conf,
        "age": feat["age"],
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
    save_history(record)
    return jsonify({"risk": risk, "probability": round(prob, 4),
                    "model": mname, "confidence": conf, "advice": advice})


@app.route("/api/analysis")
def api_analysis():
    df = pd.read_csv(DATA_PATH)
    return jsonify({
        "metrics"          : results_cache,
        "best_model"       : best_model_name,
        "chart_comparison" : chart_model_comparison(),
        "chart_roc"        : chart_roc_curves(),
        "chart_fi"         : chart_feature_importance(),
        "chart_dist"       : chart_risk_distribution(df),
    })


@app.route("/api/cm")
def api_cm():
    model_name = request.args.get("model", "Random Forest")
    return jsonify({"chart": chart_confusion_matrix(model_name)})


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    with open(HISTORY_PATH, "w") as f: json.dump([], f)
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CHAMKI BHUKHAR PREDICTION SYSTEM — MAJOR PROJECT")
    print("  Govt. Polytechnic Motihari | CSE Dept.")
    print("=" * 60)

    # Step 1 — Dataset
    if not os.path.exists(DATA_PATH):
        df = generate_dataset(n=1200)
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[✓] Dataset loaded: {len(df)} rows from {DATA_PATH}")

    # Step 2 — Train all models
    train_all_models(df)

    # Step 3 — Launch Flask
    print("\n[🌐] Starting Web App → http://localhost:5000")
    print("[ℹ]  Ctrl+C se band karo\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
