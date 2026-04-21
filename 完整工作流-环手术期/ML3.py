from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ===== 参数 =====
SEED = 15
CV_FOLDS = 10
TARGET_COLUMN = "PulmonaryInfection"

BASE_INPUT_DIR = Path("outputs/model_analysis")
TRAIN_PATH = BASE_INPUT_DIR / "train_set.csv"
TEST_PATH = BASE_INPUT_DIR / "test_set.csv"
FEATURES_PATH = BASE_INPUT_DIR / "independent_predictors.json"

OUTPUT_DIR = Path("outputs/ml_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 工具函数 =====
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    features = json.loads(FEATURES_PATH.read_text())["independent_predictors"]
    return train_df, test_df, features

def prepare_xy(df, features):
    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET_COLUMN].astype(int)
    return X, y

# ===== 阈值优化 =====
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 100)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ===== 指标计算 =====
def calculate_metrics(y_true, y_prob):
    t = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp+fn) else 0
    specificity = tn / (tn + fp) if (tn+fp) else 0

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Threshold": t,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Youden": sensitivity + specificity - 1,
    }

# ===== 模型 =====
def get_models_and_params(pos, neg):
    models = {
        "GLM": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000),
        "RF": RandomForestClassifier(class_weight="balanced", random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "NNET": MLPClassifier(max_iter=1500, early_stopping=True),
    }

    params = {
        "GLM": {"classifier__C": [0.01, 0.1, 1, 10]},
        "RF": {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [5, 10],
            "classifier__min_samples_split": [5, 10],
            "classifier__min_samples_leaf": [2, 5],
        },
        "SVM": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", 0.1],
        },
        "NNET": {
            "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "classifier__alpha": [0.0001, 0.001],
        }
    }

    # XGB
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=neg/pos,
        tree_method="hist",
        random_state=SEED
    )

    xgb_params = {
        "colsample_bytree": [0.6, 0.8],
        "subsample": [0.6, 0.8],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [100, 300],
        "max_depth": [3, 4],
        "min_child_weight": [3, 5],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 5],
    }

    return models, params, xgb_model, xgb_params

# ===== 主训练函数 =====
def train_all_models(X_train, y_train, X_test, y_test):
    pos = sum(y_train)
    neg = len(y_train) - pos

    models, params, xgb_model, xgb_params = get_models_and_params(pos, neg)

    results = {}
    fitted_models = {}

    # ===== sklearn models =====
    for name in models:
        print(f"\n训练 {name}")

        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=SEED)),
            ("classifier", models[name])
        ])

        grid = GridSearchCV(pipe, params[name],
                            cv=CV_FOLDS,
                            scoring="average_precision",
                            n_jobs=-1)

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        probs = best_model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, probs)

        metrics["CV_AUC"] = cross_val_score(
            best_model, X_train, y_train,
            cv=CV_FOLDS, scoring="roc_auc"
        ).mean()

        results[name] = metrics
        fitted_models[name] = best_model

        print(f"{name} 完成 | AUC={metrics['AUC']:.3f} | F1={metrics['F1']:.3f}")

    # ===== XGB =====
    print("\n训练 XGB")

    grid = GridSearchCV(
        xgb_model,
        xgb_params,
        cv=CV_FOLDS,
        scoring="average_precision",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    probs = best_model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, probs)

    metrics["CV_AUC"] = cross_val_score(
        best_model, X_train, y_train,
        cv=CV_FOLDS, scoring="roc_auc"
    ).mean()

    results["XGB"] = metrics
    fitted_models["XGB"] = best_model

    print(f"XGB 完成 | AUC={metrics['AUC']:.3f} | F1={metrics['F1']:.3f}")

    return results, fitted_models

# ===== 主程序 =====
def main():
    train_df, test_df, features = load_data()

    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)

    results, models = train_all_models(X_train, y_train, X_test, y_test)

    df = pd.DataFrame(results).T
    df = df.sort_values("AUC", ascending=False)

    print("\n=== 最终结果 ===")
    print(df)

    df.to_csv(OUTPUT_DIR / "final_metrics.csv")

    # ===== SHAP（最佳模型）=====
    best_name = df.index[0]
    best_model = models[best_name]

    if hasattr(best_model, "named_steps"):
        best_model = best_model.named_steps["classifier"]

    print(f"\nSHAP 分析模型: {best_name}")

    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()