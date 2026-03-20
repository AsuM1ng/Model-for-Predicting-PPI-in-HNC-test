"""基于 LASSO + 相关性过滤后的训练集特征开展多因素 Logistic 回归。

流程：
1. 读取 `lasso.py` 生成的训练集和最终特征列表；
2. 仅在训练集中拟合多因素 Logistic 回归；
3. 输出各变量 OR、95%CI、P 值；
4. 保存独立预测因素结果，供机器学习建模脚本复用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTPUT_DIR = Path('analysis_outputs')
TRAIN_PATH = OUTPUT_DIR / 'train_set.csv'
FILTERED_FEATURES_PATH = OUTPUT_DIR / 'selected_features_after_correlation.json'
LOGISTIC_RESULTS_PATH = OUTPUT_DIR / 'multivariable_logistic_results.csv'
INDEPENDENT_FEATURES_PATH = OUTPUT_DIR / 'independent_predictors.json'
TARGET_COLUMN = 'PulmonaryInfection'
P_VALUE_THRESHOLD = 0.05


def load_inputs() -> tuple[pd.DataFrame, list[str]]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError('未找到 analysis_outputs/train_set.csv，请先运行 lasso.py。')
    if not FILTERED_FEATURES_PATH.exists():
        raise FileNotFoundError('未找到 selected_features_after_correlation.json，请先运行 lasso.py。')

    train_df = pd.read_csv(TRAIN_PATH)
    with FILTERED_FEATURES_PATH.open('r', encoding='utf-8') as file:
        payload = json.load(file)
    features = payload.get('final_features', [])
    if not features:
        raise ValueError('相关性过滤后的特征列表为空，无法进行多因素 Logistic 回归。')
    return train_df, features


def prepare_design_matrix(train_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [column for column in [*features, TARGET_COLUMN] if column not in train_df.columns]
    if missing_columns:
        raise KeyError(f'训练集中缺少以下列：{missing_columns}')

    X = train_df[features].copy()
    for column in X.columns:
        if X[column].dtype == 'object' or str(X[column].dtype).startswith('category'):
            X[column] = pd.factorize(X[column].astype(str))[0]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = train_df[TARGET_COLUMN].astype(int)
    return X, y


def fit_multivariable_logistic(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_with_const = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X_with_const)
    result = model.fit(disp=False, maxiter=200)

    confidence_intervals = result.conf_int()
    records: list[dict[str, float | str]] = []
    for column in X.columns:
        coefficient = float(result.params[column])
        lower_ci = float(confidence_intervals.loc[column, 0])
        upper_ci = float(confidence_intervals.loc[column, 1])
        p_value = float(result.pvalues[column])
        records.append({
            'feature': column,
            'coefficient': coefficient,
            'odds_ratio': float(np.exp(coefficient)),
            'p_value': p_value,
            'or_95ci_lower': float(np.exp(lower_ci)),
            'or_95ci_upper': float(np.exp(upper_ci)),
            'is_independent_predictor': p_value < P_VALUE_THRESHOLD,
        })

    return pd.DataFrame(records).sort_values(['p_value', 'feature'], ascending=[True, True])


def main() -> None:
    train_df, features = load_inputs()
    X, y = prepare_design_matrix(train_df, features)
    results_df = fit_multivariable_logistic(X, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(LOGISTIC_RESULTS_PATH, index=False, encoding='utf-8-sig')

    independent_predictors = results_df.loc[
        results_df['is_independent_predictor'], 'feature'
    ].tolist()
    with INDEPENDENT_FEATURES_PATH.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'target': TARGET_COLUMN,
                'p_value_threshold': P_VALUE_THRESHOLD,
                'independent_predictors': independent_predictors,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print('=== 多因素 Logistic 回归完成 ===')
    print(results_df.to_string(index=False))
    print(f'独立预测因素 ({len(independent_predictors)} 个): {independent_predictors}')


if __name__ == '__main__':
    main()
