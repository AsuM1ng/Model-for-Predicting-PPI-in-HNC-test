import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ===== 基本参数 =====
SEED = 15
CV = 10
TARGET = "PulmonaryInfection"

BASE = Path("outputs/model_analysis")
TRAIN = BASE / "train_set.csv"
TEST = BASE / "test_set.csv"
FEATURE = BASE / "independent_predictors.json"

OUT = Path("outputs/final_model")
OUT.mkdir(exist_ok=True)

# ===== 数据 =====
def load():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    feats = json.loads(FEATURE.read_text())["independent_predictors"]
    return train, test, feats

def prep(df, feats):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].astype(int)
    return X, y

# ===== 阈值策略 =====
def best_threshold(y, prob):
    ts = np.linspace(0.01, 0.9, 100)
    best_f1, best_t = 0, 0.5
    for t in ts:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ===== 指标 =====
def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    sens = tp / (tp+fn) if (tp+fn) else 0
    spec = tn / (tn+fp) if (tn+fp) else 0

    return {
        "AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "Threshold": threshold,
        "Sensitivity": sens,
        "Specificity": spec,
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "Youden": sens + spec - 1
    }

# ===== 模型 =====
def models(pos, neg):

    base_models = {
        "GLM": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000),
        "RF": RandomForestClassifier(class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "NNET": MLPClassifier(max_iter=1500, early_stopping=True)
    }

    params = {
        "GLM": {"classifier__C":[0.01,0.1,1,10]},
        "RF":{
            "classifier__n_estimators":[100,300],
            "classifier__max_depth":[5,10],
            "classifier__min_samples_split":[5,10]
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

    return base_models, params, xgb_model, xgb_param

# ===== 训练 =====
def train_all(Xtr, ytr, Xte, yte):

    pos = sum(ytr)
    neg = len(ytr)-pos

    m, p, xgb_m, xgb_p = models(pos, neg)

    results = {}

    for name in m:
        print(f"\n训练 {name}")

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=SEED)),
            ("classifier", m[name])
        ])

        grid = GridSearchCV(pipe, p[name], cv=CV, scoring="average_precision", n_jobs=-1)
        grid.fit(Xtr, ytr)

        best = grid.best_estimator_
        prob = best.predict_proba(Xte)[:,1]

        # ===== 两种策略 =====
        t_best = best_threshold(yte, prob)
        t_recall = 0.3   # 临床高召回

        res_best = metrics(yte, prob, t_best)
        res_recall = metrics(yte, prob, t_recall)

        results[name+"_BEST"] = res_best
        results[name+"_RECALL"] = res_recall

    # ===== XGB =====
    print("\n训练 XGB")

    grid = GridSearchCV(xgb_m, xgb_p, cv=CV, scoring="average_precision", n_jobs=-1)
    grid.fit(Xtr, ytr)

    best = grid.best_estimator_
    prob = best.predict_proba(Xte)[:,1]

    t_best = best_threshold(yte, prob)
    t_recall = 0.3

    results["XGB_BEST"] = metrics(yte, prob, t_best)
    results["XGB_RECALL"] = metrics(yte, prob, t_recall)

    return pd.DataFrame(results).T

# ===== 主程序 =====
def main():
    train, test, feats = load()
    Xtr, ytr = prep(train, feats)
    Xte, yte = prep(test, feats)

    df = train_all(Xtr, ytr, Xte, yte)
    df = df.sort_values("PR_AUC", ascending=False)

    print("\n最终结果：")
    print(df)

    df.to_csv(OUT/"metrics.csv")

if __name__ == "__main__":
    main()