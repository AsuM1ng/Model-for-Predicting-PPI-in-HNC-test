# ================== 防Tkinter报错（必须最前） ==================
import matplotlib
matplotlib.use('Agg')

# ================== IMPORT ==================
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ================== 全局绘图风格 ==================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 300
})


# ================== PATH ==================
BASE = Path("outputs/model_analysis")
TRAIN_PATH = BASE / "train_set.csv"
TEST_PATH = BASE / "test_set.csv"
FEATURE_PATH = BASE / "independent_predictors.json"

OUT = Path("outputs/final_paper_ready")
OUT.mkdir(parents=True, exist_ok=True)


# ================== 参数 ==================
SEED = 15
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

THRESHOLDS = {
    "GLM": 0.4,
    "RF": 0.4,
    "SVM": 0.4,
    "NNET": 0.4,
    "XGB": 0.4
}


# ================== 数据 ==================
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feats = json.loads(FEATURE_PATH.read_text())["independent_predictors"]
    return train, test, feats

def prepare(df, feats):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = df["PulmonaryInfection"].astype(int)
    return X, y


# ================== AUC CI ==================
def auc_ci(y_true, y_prob, n_bootstraps=1000):
    rng = np.random.RandomState(SEED)
    scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ================== 指标 ==================
def calc_metrics(y, prob, t):

    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0

    auc = roc_auc_score(y, prob)
    ci_l, ci_u = auc_ci(y, prob)

    return {
        "AUC": auc,
        "AUC_CI": f"{ci_l:.3f}-{ci_u:.3f}",
        "PR_AUC": average_precision_score(y, prob),
        "Threshold": t,
        "Sensitivity": sens,
        "Specificity": spec,
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "Youden": sens + spec - 1
    }


# ================== 模型 ==================
def get_models(pos, neg):

    models = {
        "GLM": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000),

        "RF": RandomForestClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=5,
            random_state=SEED
        ),

        "SVM": SVC(probability=True, class_weight="balanced"),

        "NNET": MLPClassifier(max_iter=1500, early_stopping=True)
    }

    params = {
        "GLM": {"classifier__C":[0.01,0.1,1]},
        "RF": {"classifier__n_estimators":[200]},
        "SVM": {"classifier__C":[0.1,1], "classifier__kernel":["linear","rbf"]},
        "NNET": {"classifier__hidden_layer_sizes":[(50,)], "classifier__alpha":[0.0001]}
    }

    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=neg/pos,
        tree_method="hist",
        random_state=SEED
    )

    xgb_param = {
        "learning_rate":[0.01,0.05],
        "max_depth":[3,4],
        "min_child_weight":[5],
        "reg_lambda":[5],
        "n_estimators":[200]
    }

    return models, params, xgb_model, xgb_param


# ================== SHAP ==================
def plot_shap(model, X, name):

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(OUT/f"SHAP_beeswarm_{name}.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(
            shap_values.values,
            X,
            plot_type="bar",
            show=False
        )
        # 强制改蓝色
        for bar in plt.gca().patches:
            bar.set_color("#1f77b4")
        plt.savefig(OUT/f"SHAP_bar_{name}.png", bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"SHAP failed for {name}: {e}")


# ================== 曲线 ==================
def plot_all_curves(y, probs):

    # ROC
    plt.figure()
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(OUT/"ROC.png", bbox_inches="tight")
    plt.close()

    # PR
    plt.figure()
    for name, prob in probs.items():
        p, r, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        plt.plot(r, p, label=f"{name} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(OUT/"PR.png", bbox_inches="tight")
    plt.close()


# ================== 训练 ==================
def train_all(Xtr, ytr, Xte, yte):

    pos = sum(ytr)
    neg = len(ytr) - pos

    models, params, xgb_model, xgb_param = get_models(pos, neg)

    results = {}
    prob_store = {}

    for name in models:

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.2, random_state=SEED)),
            ("classifier", models[name])
        ])

        grid = GridSearchCV(pipe, params[name], cv=CV,
                            scoring="average_precision", n_jobs=-1)

        grid.fit(Xtr, ytr)
        best = grid.best_estimator_

        prob = best.predict_proba(Xte)[:,1]
        prob_store[name] = prob

        res = calc_metrics(yte, prob, THRESHOLDS[name])
        res["CV_AUC"] = cross_val_score(best, Xtr, ytr, cv=CV,
                                        scoring="roc_auc").mean()

        results[name] = res

        plot_shap(best.named_steps["classifier"], Xtr, name)

    # XGB
    grid = GridSearchCV(xgb_model, xgb_param, cv=CV,
                        scoring="average_precision", n_jobs=-1)
    grid.fit(Xtr, ytr)

    best = grid.best_estimator_
    prob = best.predict_proba(Xte)[:,1]
    prob_store["XGB"] = prob

    res = calc_metrics(yte, prob, THRESHOLDS["XGB"])
    res["CV_AUC"] = cross_val_score(best, Xtr, ytr, cv=CV,
                                    scoring="roc_auc").mean()

    results["XGB"] = res
    plot_shap(best, Xtr, "XGB")

    return pd.DataFrame(results).T, prob_store


# ================== MAIN ==================
def main():

    train, test, feats = load_data()
    Xtr, ytr = prepare(train, feats)
    Xte, yte = prepare(test, feats)

    df, probs = train_all(Xtr, ytr, Xte, yte)

    df = df.sort_values("PR_AUC", ascending=False)

    print("\n===== FINAL RESULTS =====")
    print(df)

    df.to_csv(OUT/"metrics.csv")

    plot_all_curves(yte, probs)


if __name__ == "__main__":
    main()