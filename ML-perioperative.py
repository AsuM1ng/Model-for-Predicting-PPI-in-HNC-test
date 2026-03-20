from __future__ import annotations

"""基于独立预测因素构建6种机器学习模型并进行验证与解释。"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("analysis_outputs")
TRAIN_PATH = OUTPUT_DIR / "train_set.csv"
TEST_PATH = OUTPUT_DIR / "test_set.csv"
INDEPENDENT_FEATURES_PATH = OUTPUT_DIR / "independent_predictors.json"
METRICS_PATH = OUTPUT_DIR / "ml_model_metrics.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "bootstrap_auc_summary.csv"
ROC_PATH = OUTPUT_DIR / "roc_curves.png"
SHAP_DIR = OUTPUT_DIR / "shap"
TARGET_COLUMN = "PulmonaryInfection"
RANDOM_STATE = 42
CV_REPEATS = 100
CV_SPLITS = 10
BOOTSTRAP_ROUNDS = 1000


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    features = json.loads(INDEPENDENT_FEATURES_PATH.read_text(encoding="utf-8")).get("independent_predictors", [])
    if not features:
        raise ValueError("独立预测因素列表为空，无法建模。")
    return train_df, test_df, features


def prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    for column in tqdm(X.columns, desc="建模输入矩阵预处理", leave=False):
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].fillna(X[column].median())
    return X, df[TARGET_COLUMN].astype(int)


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "GLM": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear"))]),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
        "NNET": Pipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.001, max_iter=3000, random_state=RANDOM_STATE))]),
        "GBM": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE)
    else:
        tqdm.write("警告：当前环境未安装xgboost，XGBoost模型将被跳过。")
    return models


def evaluate_threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "test_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred)),
    }


def bootstrap_auc_ci(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_idx = np.arange(len(y_train))
    auc_values: list[float] = []
    with tqdm(total=BOOTSTRAP_ROUNDS, desc="Bootstrap内部验证", leave=False) as pbar:
        while len(auc_values) < BOOTSTRAP_ROUNDS:
            sample_indices = rng.choice(train_idx, size=len(train_idx), replace=True)
            if len(np.unique(y_train.iloc[sample_indices])) < 2:
                continue
            sampled_model = clone(model)
            sampled_model.fit(X_train.iloc[sample_indices], y_train.iloc[sample_indices])
            probs = sampled_model.predict_proba(X_test)[:, 1]
            auc_values.append(float(roc_auc_score(y_test, probs)))
            pbar.update(1)
    lower, median, upper = np.percentile(auc_values, [2.5, 50, 97.5])
    return float(lower), float(median), float(upper)


def explain_model(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    sample_train = X_train.sample(min(80, len(X_train)), random_state=RANDOM_STATE)
    sample_test = X_test.sample(min(80, len(X_test)), random_state=RANDOM_STATE)
    classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    scaler = model.named_steps.get("scaler") if isinstance(model, Pipeline) else None
    transformed_train = pd.DataFrame(scaler.transform(sample_train), columns=sample_train.columns) if scaler is not None else sample_train
    transformed_test = pd.DataFrame(scaler.transform(sample_test), columns=sample_test.columns) if scaler is not None else sample_test
    try:
        if model_name in {"RF", "GBM", "XGBoost"}:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        elif model_name == "GLM":
            explainer = shap.LinearExplainer(classifier, transformed_train)
            shap_values = explainer.shap_values(transformed_test)
        else:
            background = transformed_train.iloc[: min(30, len(transformed_train))]
            target_eval = transformed_test.iloc[: min(30, len(transformed_test))]
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(target_eval, nsamples=100)
            transformed_test = target_eval
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        plt.figure()
        shap.summary_plot(shap_values, transformed_test, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure()
        shap.summary_plot(shap_values, transformed_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        (SHAP_DIR / f"{model_name}_shap_fallback.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train_df, test_df, features = load_datasets()
    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)
    models = build_models()
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_records: list[dict[str, float | str]] = []
    bootstrap_records: list[dict[str, float | str]] = []
    roc_payload: dict[str, dict[str, list[float]]] = {}

    for model_name, model in tqdm(models.items(), desc="模型训练与评估"):
        tqdm.write(f"开始处理模型：{model_name}")
        cv_scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=None)
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_threshold_metrics(y_test, y_prob)
        auc_lower, auc_median, auc_upper = bootstrap_auc_ci(fitted_model, X_train, y_train, X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_payload[model_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics_records.append({
            "model": model_name,
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores, ddof=1)),
            **metrics,
            "bootstrap_auc_ci_lower": auc_lower,
            "bootstrap_auc_ci_median": auc_median,
            "bootstrap_auc_ci_upper": auc_upper,
            "n_features": len(features),
        })
        bootstrap_records.append({"model": model_name, "auc_ci_lower": auc_lower, "auc_ci_median": auc_median, "auc_ci_upper": auc_upper})
        explain_model(model_name, fitted_model, X_train, X_test)

    metrics_df = pd.DataFrame(metrics_records).sort_values("cv_auc_mean", ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_records).to_csv(BOOTSTRAP_PATH, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    for model_name, roc_data in roc_payload.items():
        plt.plot(roc_data["fpr"], roc_data["tpr"], label=model_name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curves on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print("=== 机器学习建模完成 ===")
    print(metrics_df.to_string(index=False))
