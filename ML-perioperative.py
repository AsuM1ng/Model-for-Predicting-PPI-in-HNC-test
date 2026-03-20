"""基于独立预测因素构建围手术期肺部感染预测模型。

模型：GLM、RF、SVM、NNET、GBM、XGBoost。
评估：AUC、Accuracy、Sensitivity、Specificity、F1。
- AUC 使用训练集上 100 次重复 10 折交叉验证的平均值；
- 测试集用于计算 Accuracy/Sensitivity/Specificity/F1 和外部留出集 AUC；
- Bootstrap 内部验证重复 1000 次，计算测试集 AUC 的置信区间；
- 对所有模型输出 SHAP 解释结果。
"""

from __future__ import annotations

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

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

OUTPUT_DIR = Path('analysis_outputs')
TRAIN_PATH = OUTPUT_DIR / 'train_set.csv'
TEST_PATH = OUTPUT_DIR / 'test_set.csv'
INDEPENDENT_FEATURES_PATH = OUTPUT_DIR / 'independent_predictors.json'
TARGET_COLUMN = 'PulmonaryInfection'
METRICS_PATH = OUTPUT_DIR / 'ml_model_metrics.csv'
ROC_PATH = OUTPUT_DIR / 'roc_curves.png'
BOOTSTRAP_PATH = OUTPUT_DIR / 'bootstrap_auc_summary.csv'
SHAP_DIR = OUTPUT_DIR / 'shap'
RANDOM_STATE = 42
CV_REPEATS = 100
CV_SPLITS = 10
BOOTSTRAP_ROUNDS = 1000


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required_paths = [TRAIN_PATH, TEST_PATH, INDEPENDENT_FEATURES_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'缺少以下输入文件，请先运行前序脚本：{missing}')

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    with INDEPENDENT_FEATURES_PATH.open('r', encoding='utf-8') as file:
        features = json.load(file).get('independent_predictors', [])
    if not features:
        raise ValueError('独立预测因素列表为空，无法构建机器学习模型。')
    return train_df, test_df, features


def prepare_xy(dataframe: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe[features].copy()
    for column in X.columns:
        if X[column].dtype == 'object' or str(X[column].dtype).startswith('category'):
            X[column] = pd.factorize(X[column].astype(str))[0]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = dataframe[TARGET_COLUMN].astype(int)
    return X, y


def build_models() -> dict[str, Pipeline | object]:
    models: dict[str, Pipeline | object] = {
        'GLM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear')),
        ]),
        'RF': RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        ),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)),
        ]),
        'NNET': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.001, max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        'GBM': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
        )
    return models


def evaluate_threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'test_auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred),
    }


def bootstrap_auc_ci(model, X_train, y_train, X_test, y_test, n_rounds: int = BOOTSTRAP_ROUNDS) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    auc_values: list[float] = []
    train_array = np.arange(len(y_train))
    y_train_np = np.asarray(y_train)

    while len(auc_values) < n_rounds:
        sample_indices = rng.choice(train_array, size=len(train_array), replace=True)
        if len(np.unique(y_train_np[sample_indices])) < 2:
            continue
        sampled_model = clone(model)
        sampled_model.fit(X_train.iloc[sample_indices], y_train.iloc[sample_indices])
        probabilities = sampled_model.predict_proba(X_test)[:, 1]
        auc_values.append(roc_auc_score(y_test, probabilities))

    lower, median, upper = np.percentile(auc_values, [2.5, 50, 97.5])
    return float(lower), float(median), float(upper)


def explain_model(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list[str]) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    sample_train = X_train.sample(min(80, len(X_train)), random_state=RANDOM_STATE)
    sample_test = X_test.sample(min(80, len(X_test)), random_state=RANDOM_STATE)

    try:
        classifier = model.named_steps['classifier'] if isinstance(model, Pipeline) else model
        transformer = model.named_steps['scaler'] if isinstance(model, Pipeline) and 'scaler' in model.named_steps else None
        transformed_train = transformer.transform(sample_train) if transformer is not None else sample_train
        transformed_test = transformer.transform(sample_test) if transformer is not None else sample_test
        transformed_train_df = pd.DataFrame(transformed_train, columns=sample_train.columns, index=sample_train.index)
        transformed_test_df = pd.DataFrame(transformed_test, columns=sample_test.columns, index=sample_test.index)

        if model_name in {'RF', 'GBM', 'XGBoost'}:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_test_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        elif model_name == 'GLM':
            explainer = shap.LinearExplainer(classifier, transformed_train_df)
            shap_values = explainer.shap_values(transformed_test_df)
        else:
            background = transformed_train_df.iloc[: min(30, len(transformed_train_df))]
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(transformed_test_df.iloc[: min(30, len(transformed_test_df))], nsamples=100)
            transformed_test_df = transformed_test_df.iloc[: min(30, len(transformed_test_df))]
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

        plt.figure()
        shap.summary_plot(shap_values, transformed_test_df, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f'{model_name}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, transformed_test_df, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f'{model_name}_shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as error:  # pragma: no cover
        fallback_path = SHAP_DIR / f'{model_name}_shap_fallback.json'
        with fallback_path.open('w', encoding='utf-8') as file:
            json.dump({'error': str(error), 'feature_names': feature_names}, file, ensure_ascii=False, indent=2)


def main() -> None:
    train_df, test_df, features = load_datasets()
    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)

    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    models = build_models()
    metrics_records: list[dict[str, float | str]] = []
    bootstrap_records: list[dict[str, float | str]] = []
    roc_payload: dict[str, dict[str, list[float]]] = {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f'>>> 训练模型: {model_name}')
        cv_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=None)
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_prob = fitted_model.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_threshold_metrics(y_test, y_prob)
        auc_lower, auc_median, auc_upper = bootstrap_auc_ci(fitted_model, X_train, y_train, X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_payload[model_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

        metrics_records.append({
            'model': model_name,
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores, ddof=1)),
            'test_auc': test_metrics['test_auc'],
            'accuracy': test_metrics['accuracy'],
            'sensitivity': test_metrics['sensitivity'],
            'specificity': test_metrics['specificity'],
            'f1': test_metrics['f1'],
            'bootstrap_auc_ci_lower': auc_lower,
            'bootstrap_auc_ci_median': auc_median,
            'bootstrap_auc_ci_upper': auc_upper,
            'n_features': len(features),
        })
        bootstrap_records.append({
            'model': model_name,
            'auc_ci_lower': auc_lower,
            'auc_ci_median': auc_median,
            'auc_ci_upper': auc_upper,
        })

        explain_model(model_name, fitted_model, X_train, X_test, features)

    metrics_df = pd.DataFrame(metrics_records).sort_values('cv_auc_mean', ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding='utf-8-sig')
    pd.DataFrame(bootstrap_records).to_csv(BOOTSTRAP_PATH, index=False, encoding='utf-8-sig')

    plt.figure(figsize=(8, 6))
    for model_name, roc_data in roc_payload.items():
        plt.plot(roc_data['fpr'], roc_data['tpr'], label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curves on Test Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print('=== 机器学习建模完成 ===')
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
