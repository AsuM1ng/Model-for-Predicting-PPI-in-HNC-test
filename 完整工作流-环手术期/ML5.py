# ================== IMPORT ==================
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import shap
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, accuracy_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ================== PATH ==================
BASE = Path("outputs/model_analysis")
TRAIN_PATH = BASE / "train_set.csv"
TEST_PATH = BASE / "test_set.csv"
FEATURE_PATH = BASE / "independent_predictors.json"

OUT = Path("outputs/final_full")
OUT.mkdir(parents=True, exist_ok=True)


# ================== DATA ==================
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = json.loads(FEATURE_PATH.read_text())["independent_predictors"]
    return train, test, features

def prepare(df, features):
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = df["PulmonaryInfection"].astype(int)
    return X, y


# ================== THRESHOLD ==================
def find_best_threshold(y, prob):
    ts = np.linspace(0.01, 0.9, 100)
    best_f1, best_t = 0, 0.5
    for t in ts:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ================== METRICS ==================
def calc_metrics(y, prob, t):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0

    return {
        "AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "Threshold": t,
        "Sensitivity": sens,
        "Specificity": spec,
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "Youden": sens + spec - 1
    }


# ================== MODELS ==================
def get_models(pos, neg):

    models = {
        "GLM": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000),
        "RF": RandomForestClassifier(class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "NNET": MLPClassifier(max_iter=1500, early_stopping=True)
    }

    params = {
        "GLM": {"classifier__C":[0.01,0.1,1,10]},
        "RF":{
            "classifier__n_estimators":[100,300],
            "classifier__max_depth":[5,10]
        },
        "SVM":{
            "classifier__C":[0.1,1,10],
            "classifier__kernel":["linear","rbf"]
        },
        "NNET":{
            "classifier__hidden_layer_sizes":[(50,),(100,)],
            "classifier__alpha":[0.0001,0.001]
        }
    }

    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=neg/pos,
        tree_method="hist"
    )

    xgb_param = {
        "learning_rate":[0.01,0.05],
        "max_depth":[3,4],
        "n_estimators":[100,300],
        "reg_lambda":[1,5]
    }

    return models, params, xgb_model, xgb_param


# ================== TRAIN ==================
def train_all(Xtr, ytr, Xte, yte):

    pos = sum(ytr)
    neg = len(ytr)-pos

    models, params, xgb_model, xgb_param = get_models(pos, neg)

    results = {}
    prob_store = {}

    for name in models:
        print(f"\n训练 {name}")

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE()),
            ("classifier", models[name])
        ])

        grid = GridSearchCV(pipe, params[name], cv=10,
                            scoring="average_precision", n_jobs=-1)

        grid.fit(Xtr, ytr)
        best = grid.best_estimator_

        prob = best.predict_proba(Xte)[:,1]

        t_best = find_best_threshold(yte, prob)
        t_recall = 0.3

        results[name+"_BEST"] = calc_metrics(yte, prob, t_best)
        results[name+"_RECALL"] = calc_metrics(yte, prob, t_recall)

        prob_store[name] = prob

    # XGB
    print("\n训练 XGB")
    grid = GridSearchCV(xgb_model, xgb_param, cv=10,
                        scoring="average_precision", n_jobs=-1)
    grid.fit(Xtr, ytr)

    prob = grid.best_estimator_.predict_proba(Xte)[:,1]

    t_best = find_best_threshold(yte, prob)

    results["XGB_BEST"] = calc_metrics(yte, prob, t_best)
    results["XGB_RECALL"] = calc_metrics(yte, prob, 0.3)

    prob_store["XGB"] = prob

    return pd.DataFrame(results).T, prob_store


# ================== PLOTS ==================
def plot_all_curves(y, prob_dict):

    plt.figure(figsize=(8,6))
    for name, prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(OUT/"roc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    for name, prob in prob_dict.items():
        p, r, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        plt.plot(r, p, label=f"{name} (AP={ap:.2f})")
    plt.legend()
    plt.title("PR Curve")
    plt.savefig(OUT/"pr.png", dpi=300)
    plt.close()


def plot_calibration(y, prob, name):
    pt, pp = calibration_curve(y, prob, n_bins=10)
    plt.figure()
    plt.plot(pp, pt, marker='o', label=name)
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("Calibration")
    plt.savefig(OUT/f"cal_{name}.png", dpi=300)
    plt.close()

def plot_dca(y, prob, name):
    ts = np.linspace(0.01,0.5,100)
    N = len(y)
    nb = []
    for t in ts:
        pred = (prob>=t).astype(int)
        TP = np.sum((pred==1)&(y==1))
        FP = np.sum((pred==1)&(y==0))
        nb.append((TP/N)-(FP/N)*(t/(1-t)))

    plt.figure()
    plt.plot(ts, nb, label=name)
    plt.plot(ts, np.zeros_like(ts), linestyle=":")
    plt.legend()
    plt.title("DCA")
    plt.savefig(OUT/f"dca_{name}.png", dpi=300)
    plt.close()


# ================== MAIN ==================
def main():

    train, test, feats = load_data()
    Xtr, ytr = prepare(train, feats)
    Xte, yte = prepare(test, feats)

    df, prob_dict = train_all(Xtr, ytr, Xte, yte)

    print("\n=== 最终结果 ===")
    print(df)

    df.to_csv(OUT/"metrics.csv")

    plot_all_curves(yte, prob_dict)

    # 选最优模型（按PR_AUC）
    best_model = df.sort_values("PR_AUC", ascending=False).index[0].split("_")[0]
    best_prob = prob_dict[best_model]

    plot_calibration(yte, best_prob, best_model)
    plot_dca(yte, best_prob, best_model)

    print(f"\n最佳模型: {best_model}")

if __name__ == "__main__":
    main()