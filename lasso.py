"""基于 data1.csv 执行分层抽样、LASSO 特征筛选与相关性过滤。

流程：
1. 读取 `data1.csv`；
2. 以 `PulmonaryInfection` 为结局进行 7:3 分层随机抽样；
3. 将全部特征选择过程限制在训练集内；
4. 使用 L1 Logistic（LASSO）筛选非零系数变量；
5. 基于训练集对入选变量做 Spearman 相关性分析，若绝对相关系数 > 0.6，删除 LASSO
   绝对系数较小的变量；
6. 保存训练集、测试集、LASSO 系数表以及过滤后的最终特征列表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path('data1.csv')
TARGET_COLUMN = 'PulmonaryInfection'
TEST_SIZE = 0.3
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.6
OUTPUT_DIR = Path('analysis_outputs')
TRAIN_PATH = OUTPUT_DIR / 'train_set.csv'
TEST_PATH = OUTPUT_DIR / 'test_set.csv'
LASSO_COEF_PATH = OUTPUT_DIR / 'lasso_coefficients.csv'
LASSO_SELECTED_PATH = OUTPUT_DIR / 'lasso_selected_features.json'
CORRELATION_MATRIX_PATH = OUTPUT_DIR / 'lasso_spearman_matrix.csv'
FILTERED_FEATURES_PATH = OUTPUT_DIR / 'selected_features_after_correlation.json'
SPLIT_SUMMARY_PATH = OUTPUT_DIR / 'split_summary.json'

EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'未找到 {DATA_PATH}，请先运行 data_clean1.py 生成清洗数据。')
    return pd.read_csv(DATA_PATH)


def prepare_xy(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f'数据中不存在结局列 {TARGET_COLUMN}。')

    feature_columns = [
        column for column in dataframe.columns
        if column not in EXCLUDED_COLUMNS
    ]
    X = dataframe[feature_columns].copy()
    y = dataframe[TARGET_COLUMN].astype(int)

    for column in X.columns:
        if X[column].dtype == 'object' or str(X[column].dtype).startswith('category'):
            X[column] = pd.factorize(X[column].astype(str))[0]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, Pipeline]:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'classifier',
            LogisticRegressionCV(
                penalty='l1',
                solver='liblinear',
                cv=10,
                scoring='roc_auc',
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=None,
                refit=True,
            ),
        ),
    ])
    pipeline.fit(X_train, y_train)
    coefficients = pipeline.named_steps['classifier'].coef_.ravel()
    coefficient_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
    }).sort_values(['abs_coefficient', 'feature'], ascending=[False, True])
    return coefficient_df, pipeline


def correlation_filter(
    X_train: pd.DataFrame,
    coefficient_df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], pd.DataFrame, list[dict[str, float | str]]]:
    selected_df = coefficient_df[coefficient_df['coefficient'] != 0].copy()
    selected_features = selected_df['feature'].tolist()
    if not selected_features:
        raise ValueError('LASSO 未筛选出任何非零系数变量，请检查数据或调整模型参数。')

    selected_train = X_train[selected_features]
    corr_matrix = selected_train.corr(method='spearman').fillna(0)
    abs_corr = corr_matrix.abs()

    coef_lookup = selected_df.set_index('feature')['abs_coefficient'].to_dict()
    to_remove: set[str] = set()
    removal_records: list[dict[str, float | str]] = []

    for index, left_feature in enumerate(selected_features):
        if left_feature in to_remove:
            continue
        for right_feature in selected_features[index + 1:]:
            if right_feature in to_remove:
                continue
            corr_value = abs_corr.loc[left_feature, right_feature]
            if corr_value > threshold:
                left_coef = coef_lookup[left_feature]
                right_coef = coef_lookup[right_feature]
                drop_feature = left_feature if left_coef < right_coef else right_feature
                keep_feature = right_feature if drop_feature == left_feature else left_feature
                to_remove.add(drop_feature)
                removal_records.append({
                    'keep_feature': keep_feature,
                    'drop_feature': drop_feature,
                    'spearman_abs': float(corr_value),
                    'keep_abs_coefficient': float(coef_lookup[keep_feature]),
                    'drop_abs_coefficient': float(coef_lookup[drop_feature]),
                })

    final_features = [feature for feature in selected_features if feature not in to_remove]
    return final_features, corr_matrix.loc[selected_features, selected_features], removal_records


def save_outputs(
    full_data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    coefficient_df: pd.DataFrame,
    final_features: list[str],
    corr_matrix: pd.DataFrame,
    removed_pairs: list[dict[str, float | str]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = full_data.loc[X_train.index].copy()
    test_df = full_data.loc[X_test.index].copy()
    train_df.to_csv(TRAIN_PATH, index=False, encoding='utf-8-sig')
    test_df.to_csv(TEST_PATH, index=False, encoding='utf-8-sig')
    coefficient_df.to_csv(LASSO_COEF_PATH, index=False, encoding='utf-8-sig')
    corr_matrix.to_csv(CORRELATION_MATRIX_PATH, encoding='utf-8-sig')

    with LASSO_SELECTED_PATH.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'selected_by_lasso': coefficient_df.loc[
                    coefficient_df['coefficient'] != 0, 'feature'
                ].tolist(),
                'random_state': RANDOM_STATE,
                'test_size': TEST_SIZE,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with FILTERED_FEATURES_PATH.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'target': TARGET_COLUMN,
                'final_features': final_features,
                'correlation_threshold': CORRELATION_THRESHOLD,
                'removed_pairs': removed_pairs,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with SPLIT_SUMMARY_PATH.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'target': TARGET_COLUMN,
                'overall_distribution': full_data[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict(),
                'train_distribution': y_train.value_counts(normalize=True).sort_index().to_dict(),
                'test_distribution': y_test.value_counts(normalize=True).sort_index().to_dict(),
                'train_size': int(len(y_train)),
                'test_size': int(len(y_test)),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    dataframe = load_dataset()
    X, y = prepare_xy(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    coefficient_df, _ = fit_lasso(X_train, y_train)
    final_features, corr_matrix, removed_pairs = correlation_filter(X_train, coefficient_df)
    save_outputs(dataframe, X_train, X_test, y_train, y_test, coefficient_df, final_features, corr_matrix, removed_pairs)

    print('=== LASSO 特征筛选完成 ===')
    print(f'训练集: {len(y_train)}，测试集: {len(y_test)}')
    print('LASSO 非零系数变量:')
    print(coefficient_df.loc[coefficient_df['coefficient'] != 0, ['feature', 'coefficient']].to_string(index=False))
    print(f'相关性过滤后保留变量 ({len(final_features)} 个): {final_features}')
    if removed_pairs:
        print('因高相关性移除的变量:')
        print(pd.DataFrame(removed_pairs).to_string(index=False))


if __name__ == '__main__':
    main()
