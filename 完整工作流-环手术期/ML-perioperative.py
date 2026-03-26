"""基于独立预测因素构建多种机器学习模型并进行调参、验证与解释。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTPUT_ROOT = Path("analysis_outputs")
TRAIN_PATH = OUTPUT_ROOT / "train_set.csv"
TEST_PATH = OUTPUT_ROOT / "test_set.csv"
INDEPENDENT_FEATURES_PATH = OUTPUT_ROOT / "independent_predictors.json"
TARGET_COLUMN = "PulmonaryInfection"
RANDOM_STATE = 42
CV_REPEATS = 10
CV_SPLITS = 10
TUNING_CV_REPEATS = 5
TUNING_CV_SPLITS = 5
THRESHOLD_CV_SPLITS = 5
TUNING_ITER = 20
BOOTSTRAP_ROUNDS = 100
THRESHOLD_GRID_SIZE = 201
MIN_SPECIFICITY_FLOOR = 0.60
N_JOBS = 1


def build_run_paths() -> dict[str, Path]:
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S_BJT")
    run_dir = OUTPUT_ROOT / f"perioperative_model_run_{timestamp}"
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "ml_model_metrics.csv",
        "bootstrap": run_dir / "bootstrap_auc_summary.csv",
        "tuning": run_dir / "ml_model_best_params.json",
        "roc": run_dir / "roc_curves.png",
        "report": run_dir / "run_summary.md",
        "shap_dir": run_dir / "shap",
    }


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
            "estimator": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
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
    candidate_count = int(np.prod([len(v) for v in param_distributions.values()]))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=min(TUNING_ITER, candidate_count),
        scoring="roc_auc",
        cv=tuning_cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    tqdm.write(f"{model_name} 最优AUC={search.best_score_:.4f}")
    return search.best_estimator_, float(search.best_score_), search.best_params_


def select_threshold_for_sensitivity(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    candidate_thresholds = np.unique(np.clip(np.concatenate(([0.0, 1.0], np.linspace(0.0, 1.0, THRESHOLD_GRID_SIZE), y_prob)), 0.0, 1.0))
    best_payload: dict[str, float] | None = None
    best_threshold = 0.5
    for threshold in candidate_thresholds:
        metrics = evaluate_threshold_metrics(y_true, y_prob, float(threshold))
        payload = {"threshold": float(threshold), **metrics}
        if best_payload is None:
            best_payload = payload
            best_threshold = float(threshold)
            continue
        current_feasible = payload["specificity"] >= MIN_SPECIFICITY_FLOOR
        best_feasible = best_payload["specificity"] >= MIN_SPECIFICITY_FLOOR
        candidate_score = (current_feasible, payload["sensitivity"], payload["f1"], payload["specificity"], payload["accuracy"], -abs(payload["threshold"] - 0.5))
        best_score = (best_feasible, best_payload["sensitivity"], best_payload["f1"], best_payload["specificity"], best_payload["accuracy"], -abs(best_payload["threshold"] - 0.5))
        if candidate_score > best_score:
            best_payload = payload
            best_threshold = float(threshold)
    if best_payload is None:
        raise ValueError("阈值搜索失败。")
    return best_threshold, best_payload


def smooth_roc_curve(fpr: np.ndarray, tpr: np.ndarray, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    dense_fpr = np.linspace(0.0, 1.0, points)
    unique_fpr, unique_idx = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_idx]
    dense_tpr = np.interp(dense_fpr, unique_fpr, unique_tpr)
    dense_tpr = np.maximum.accumulate(dense_tpr)
    dense_tpr[0] = 0.0
    dense_tpr[-1] = 1.0
    return dense_fpr, np.clip(dense_tpr, 0.0, 1.0)


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


def explain_model(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, shap_dir: Path) -> None:
    tqdm.write(f"开始进行 {model_name} 的SHAP解释...")
    shap_dir.mkdir(parents=True, exist_ok=True)
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
        plt.savefig(shap_dir / f"{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure()
        shap.summary_plot(shap_matrix, transformed_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{model_name}_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        (shap_dir / f"{model_name}_shap_fallback.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_report(run_paths: dict[str, Path], features: list[str], metrics_df: pd.DataFrame, tuning_records: dict[str, dict[str, object]]) -> None:
    report_lines = [
        "# 建模结果信息文档",
        "",
        f"- 运行目录: `{run_paths['run_dir']}`",
        f"- 训练集路径: `{TRAIN_PATH}`",
        f"- 测试集路径: `{TEST_PATH}`",
        f"- 特征数量: {len(features)}",
        f"- 特征列表: {', '.join(features)}",
        "",
        "## 关键参数",
        f"- RANDOM_STATE: {RANDOM_STATE}",
        f"- CV_SPLITS: {CV_SPLITS}",
        f"- CV_REPEATS: {CV_REPEATS}",
        f"- TUNING_CV_SPLITS: {TUNING_CV_SPLITS}",
        f"- TUNING_CV_REPEATS: {TUNING_CV_REPEATS}",
        f"- TUNING_ITER: {TUNING_ITER}",
        f"- THRESHOLD_CV_SPLITS: {THRESHOLD_CV_SPLITS}",
        f"- BOOTSTRAP_ROUNDS: {BOOTSTRAP_ROUNDS}",
        f"- THRESHOLD_GRID_SIZE: {THRESHOLD_GRID_SIZE}",
        f"- MIN_SPECIFICITY_FLOOR: {MIN_SPECIFICITY_FLOOR}",
        f"- N_JOBS: {N_JOBS}",
        "",
        "## 输出文件",
        f"- 指标汇总: `{run_paths['metrics'].name}`",
        f"- Bootstrap汇总: `{run_paths['bootstrap'].name}`",
        f"- 最佳参数: `{run_paths['tuning'].name}`",
        f"- ROC图: `{run_paths['roc'].name}`",
        f"- SHAP目录: `{run_paths['shap_dir'].name}`",
        "",
        "## 模型结果预览",
        metrics_df.to_markdown(index=False),
        "",
        "## 各模型最佳参数",
    ]
    for model_name, payload in tuning_records.items():
        report_lines.append(f"### {model_name}")
        report_lines.append(f"- best_cv_auc: {payload['best_cv_auc']:.6f}")
        report_lines.append(f"- best_params: `{json.dumps(payload['best_params'], ensure_ascii=False, sort_keys=True)}`")
        report_lines.append("")
    run_paths["report"].write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    run_paths = build_run_paths()
    run_paths["shap_dir"].mkdir(parents=True, exist_ok=True)

    with tqdm(total=5, desc="ML-perioperative总进度") as progress:
        train_df, test_df, features = load_datasets()
        progress.update(1)

        X_train, y_train = prepare_xy(train_df, features)
        X_test, y_test = prepare_xy(test_df, features)
        progress.update(1)

        model_specs = build_model_specs()
        cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        progress.update(1)

        metrics_records: list[dict[str, float | str]] = []
        bootstrap_records: list[dict[str, float | str]] = []
        tuning_records: dict[str, dict[str, object]] = {}
        roc_payload: dict[str, dict[str, list[float]]] = {}

        for model_name, spec in tqdm(model_specs.items(), desc="模型训练、调参与评估"):
            tqdm.write(f"开始处理模型：{model_name}")
            tuned_model, tuning_auc, best_params = tune_model(model_name, spec["estimator"], spec["param_distributions"], X_train, y_train)
            cv_scores = cross_val_score(tuned_model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=N_JOBS)
            threshold_cv = StratifiedKFold(n_splits=THRESHOLD_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            oof_prob = cross_val_predict(tuned_model, X_train, y_train, cv=threshold_cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
            best_threshold, threshold_payload = select_threshold_for_sensitivity(y_train, oof_prob)
            fitted_model = clone(tuned_model)
            fitted_model.fit(X_train, y_train)
            y_prob = fitted_model.predict_proba(X_test)[:, 1]
            metrics = evaluate_threshold_metrics(y_test, y_prob, threshold=best_threshold)
            auc_lower, auc_median, auc_upper = bootstrap_auc_ci(fitted_model, X_train, y_train, X_test, y_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            smooth_fpr, smooth_tpr = smooth_roc_curve(fpr, tpr)
            roc_payload[model_name] = {"fpr": smooth_fpr.tolist(), "tpr": smooth_tpr.tolist()}
            tuning_records[model_name] = {"best_cv_auc": tuning_auc, "best_params": best_params, "best_threshold": best_threshold, "threshold_selection_metrics": threshold_payload}
            metrics_records.append({
                "model": model_name,
                "selected_threshold": best_threshold,
                "threshold_train_sensitivity": threshold_payload["sensitivity"],
                "threshold_train_specificity": threshold_payload["specificity"],
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
            explain_model(model_name, fitted_model, X_train, X_test, run_paths["shap_dir"])
        progress.update(1)

        metrics_df = pd.DataFrame(metrics_records).sort_values(["test_auc", "cv_auc_mean"], ascending=False)
        metrics_df.to_csv(run_paths["metrics"], index=False, encoding="utf-8-sig")
        pd.DataFrame(bootstrap_records).to_csv(run_paths["bootstrap"], index=False, encoding="utf-8-sig")
        run_paths["tuning"].write_text(json.dumps(tuning_records, ensure_ascii=False, indent=2), encoding="utf-8")

        plt.figure(figsize=(8, 6))
        for model_name, roc_data in roc_payload.items():
            plt.plot(roc_data["fpr"], roc_data["tpr"], label=model_name)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC Curves on Test Set")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_paths["roc"], dpi=300, bbox_inches="tight")
        plt.close()
        write_run_report(run_paths, features, metrics_df, tuning_records)
        progress.update(1)

    print("=== 机器学习建模完成 ===")
    print(f"输出目录：{run_paths['run_dir']}")
    print(metrics_df.to_string(index=False))
