from __future__ import annotations

"""基于独立预测因素构建6种机器学习模型并进行验证、调参与解释。"""

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
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
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
TUNING_PATH = OUTPUT_DIR / "ml_model_best_params.json"
ROC_PATH = OUTPUT_DIR / "roc_curves.png"
SHAP_DIR = OUTPUT_DIR / "shap"
TARGET_COLUMN = "PulmonaryInfection"
RANDOM_STATE = 42
CV_REPEATS = 100
CV_SPLITS = 10
TUNING_CV_REPEATS = 5
TUNING_CV_SPLITS = 5
TUNING_ITER = 20
BOOTSTRAP_ROUNDS = 1000


def normalize_shap_values(shap_values) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    if shap_values.ndim != 2:
        raise ValueError(f"无法解析SHAP输出维度: {shap_values.shape}")
    return shap_values


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


def build_model_specs() -> dict[str, dict[str, object]]:
    model_specs: dict[str, dict[str, object]] = {
        "GLM": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)),
            ]),
            "param_distributions": {
                "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
            },
        },
        "RF": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            "param_distributions": {
                "n_estimators": [300, 500, 800, 1000],
                "max_depth": [3, 5, 8, 10, None],
                "min_samples_leaf": [1, 2, 3, 5, 10],
                "min_samples_split": [2, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5, 0.8],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "SVM": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            "param_distributions": {
                "classifier__C": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0],
                "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        },
        "NNET": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(max_iter=5000, early_stopping=True, random_state=RANDOM_STATE)),
            ]),
            "param_distributions": {
                "classifier__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)],
                "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "classifier__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
                "classifier__activation": ["relu", "tanh"],
                "classifier__solver": ["adam", "lbfgs"],
                "classifier__batch_size": [16, 32, 64],
            },
        },
        "GBM": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_distributions": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_leaf": [1, 3, 5, 10],
            },
        },
    }
    if XGBClassifier is not None:
        model_specs["XGBoost"] = {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
            ),
            "param_distributions": {
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0.0, 0.1, 0.3, 1.0],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 2.0, 5.0],
            },
        }
    else:
        tqdm.write("警告：当前环境未安装xgboost，XGBoost模型将被跳过。")
    return model_specs


def tune_model(model_name: str, estimator, param_distributions: dict[str, list[object]], X_train: pd.DataFrame, y_train: pd.Series):
    tqdm.write(f"开始调参：{model_name}")
    tuning_cv = RepeatedStratifiedKFold(n_splits=TUNING_CV_SPLITS, n_repeats=TUNING_CV_REPEATS, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=min(TUNING_ITER, int(np.prod([len(v) for v in param_distributions.values()]))),
        scoring="roc_auc",
        cv=tuning_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tqdm.write(f"{model_name} 最优AUC={search.best_score_:.4f}")
    return search.best_estimator_, float(search.best_score_), search.best_params_


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
    tqdm.write(f"开始进行 {model_name} 的SHAP解释...")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    sample_train = X_train.sample(min(80, len(X_train)), random_state=RANDOM_STATE)
    sample_test = X_test.sample(min(80, len(X_test)), random_state=RANDOM_STATE)
    classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    scaler = model.named_steps.get("scaler") if isinstance(model, Pipeline) else None
    transformed_train = pd.DataFrame(scaler.transform(sample_train), columns=sample_train.columns, index=sample_train.index) if scaler is not None else sample_train
    transformed_test = pd.DataFrame(scaler.transform(sample_test), columns=sample_test.columns, index=sample_test.index) if scaler is not None else sample_test
    try:
        if model_name in {"RF", "GBM", "XGBoost"}:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_test)
        elif model_name == "GLM":
            explainer = shap.LinearExplainer(classifier, transformed_train)
            shap_values = explainer.shap_values(transformed_test)
        else:
            background = transformed_train.iloc[: min(30, len(transformed_train))]
            target_eval = transformed_test.iloc[: min(30, len(transformed_test))]
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(target_eval, nsamples=100)
            transformed_test = target_eval
        shap_matrix = normalize_shap_values(shap_values)
        plt.figure()
        shap.summary_plot(shap_matrix, transformed_test, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure()
        shap.summary_plot(shap_matrix, transformed_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{model_name}_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        (SHAP_DIR / f"{model_name}_shap_fallback.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with tqdm(total=5, desc="ML-perioperative总进度") as progress:
        train_df, test_df, features = load_datasets()
        progress.update(1)

        X_train, y_train = prepare_xy(train_df, features)
        X_test, y_test = prepare_xy(test_df, features)
        progress.update(1)

        model_specs = build_model_specs()
        cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        progress.update(1)

        metrics_records: list[dict[str, float | str]] = []
        bootstrap_records: list[dict[str, float | str]] = []
        tuning_records: dict[str, dict[str, object]] = {}
        roc_payload: dict[str, dict[str, list[float]]] = {}

        for model_name, spec in tqdm(model_specs.items(), desc="模型训练、调参与评估"):
            tqdm.write(f"开始处理模型：{model_name}")
            tuned_model, tuning_auc, best_params = tune_model(model_name, spec["estimator"], spec["param_distributions"], X_train, y_train)
            cv_scores = cross_val_score(tuned_model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
            fitted_model = clone(tuned_model)
            fitted_model.fit(X_train, y_train)
            y_prob = fitted_model.predict_proba(X_test)[:, 1]
            metrics = evaluate_threshold_metrics(y_test, y_prob)
            auc_lower, auc_median, auc_upper = bootstrap_auc_ci(fitted_model, X_train, y_train, X_test, y_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_payload[model_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            tuning_records[model_name] = {"best_cv_auc": tuning_auc, "best_params": best_params}
            metrics_records.append({
                "model": model_name,
                "tuning_auc": tuning_auc,
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
        progress.update(1)

        metrics_df = pd.DataFrame(metrics_records).sort_values(["test_auc", "cv_auc_mean"], ascending=False)
        metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame(bootstrap_records).to_csv(BOOTSTRAP_PATH, index=False, encoding="utf-8-sig")
        TUNING_PATH.write_text(json.dumps(tuning_records, ensure_ascii=False, indent=2), encoding="utf-8")

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
        progress.update(1)

    print("=== 机器学习建模完成 ===")
    print(metrics_df.to_string(index=False))
