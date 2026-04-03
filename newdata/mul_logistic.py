from __future__ import annotations

"""训练集内多因素Logistic分析，筛选独立预测因素。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

OUTPUT_DIR = Path("analysis_outputs")
TRAIN_PATH = OUTPUT_DIR / "train_set.csv"
FILTERED_FEATURES_PATH = OUTPUT_DIR / "selected_features_after_correlation.json"
LOGISTIC_RESULTS_PATH = OUTPUT_DIR / "multivariable_logistic_results.csv"
INDEPENDENT_FEATURES_PATH = OUTPUT_DIR / "independent_predictors.json"
TARGET_COLUMN = "PulmonaryInfection"
P_VALUE_THRESHOLD = 0.05


def load_inputs() -> tuple[pd.DataFrame, list[str]]:
    train_df = pd.read_csv(TRAIN_PATH)
    features = json.loads(FILTERED_FEATURES_PATH.read_text(encoding="utf-8")).get("final_features", [])
    if not features:
        raise ValueError("相关性过滤后的特征为空。")
    return train_df, features


def prepare_design_matrix(train_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = train_df[features].copy()
    for column in tqdm(X.columns, desc="Logistic输入矩阵预处理", leave=False):
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].fillna(X[column].median())

    # 避免奇异矩阵：剔除常数列、完全重复列，以及线性相关导致秩亏的列。
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        tqdm.write(f"移除常数特征({len(constant_cols)}): {constant_cols}")
        X = X.drop(columns=constant_cols)

    duplicate_mask = X.T.duplicated(keep="first")
    duplicate_cols = X.columns[duplicate_mask].tolist()
    if duplicate_cols:
        tqdm.write(f"移除重复特征({len(duplicate_cols)}): {duplicate_cols}")
        X = X.loc[:, ~duplicate_mask]

    # 迭代删除线性相关列，直到设计矩阵满秩。
    while X.shape[1] > 1:
        matrix = X.to_numpy(dtype=float)
        rank = np.linalg.matrix_rank(matrix)
        if rank == X.shape[1]:
            break

        redundant_col = None
        for col in X.columns:
            rank_without_col = np.linalg.matrix_rank(X.drop(columns=[col]).to_numpy(dtype=float))
            if rank_without_col == rank:
                redundant_col = col
                break

        # 理论上总能找到；若未找到则兜底删除最后一列以打破秩亏。
        redundant_col = redundant_col or X.columns[-1]
        tqdm.write(f"移除线性相关特征: {redundant_col}")
        X = X.drop(columns=[redundant_col])

    if X.empty:
        raise ValueError("可用于Logistic回归的特征为空，请检查上游特征筛选结果。")

    y = train_df[TARGET_COLUMN].astype(int)
    return X, y


def fit_multivariable_logistic(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    tqdm.write("开始执行多因素Logistic回归...")
    design_matrix = sm.add_constant(X, has_constant="add")
    try:
        result = sm.Logit(y, design_matrix).fit(disp=False, maxiter=200)
    except np.linalg.LinAlgError:
        tqdm.write("检测到奇异矩阵，改用GLM(Binomial)进行稳健拟合。")
        result = sm.GLM(y, design_matrix, family=sm.families.Binomial()).fit(maxiter=200)
    conf = result.conf_int()
    records: list[dict[str, float | str | bool]] = []
    for feature in tqdm(X.columns, desc="汇总Logistic结果", leave=False):
        beta = float(result.params[feature])
        lower = float(conf.loc[feature, 0])
        upper = float(conf.loc[feature, 1])
        p_value = 0.5*float(result.pvalues[feature])
        records.append({
            "feature": feature,
            "coefficient": beta,
            "odds_ratio": float(np.exp(beta)),
            "or_95ci_lower": float(np.exp(lower)),
            "or_95ci_upper": float(np.exp(upper)),
            "p_value": p_value,
            "is_independent_predictor": p_value < P_VALUE_THRESHOLD,
        })
    return pd.DataFrame(records).sort_values(["p_value", "feature"], ascending=[True, True])


if __name__ == "__main__":
    with tqdm(total=4, desc="mul_logistic总进度") as progress:
        train_df, features = load_inputs()
        progress.update(1)

        X, y = prepare_design_matrix(train_df, features)
        progress.update(1)

        results_df = fit_multivariable_logistic(X, y)
        progress.update(1)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(LOGISTIC_RESULTS_PATH, index=False, encoding="utf-8-sig")
        independent_predictors = results_df.loc[results_df["is_independent_predictor"], "feature"].tolist()
        INDEPENDENT_FEATURES_PATH.write_text(json.dumps({
            "target": TARGET_COLUMN,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "independent_predictors": independent_predictors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        progress.update(1)

    print("=== 多因素Logistic回归完成 ===")
    print(results_df.to_string(index=False))
