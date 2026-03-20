"""基于清洗后的围手术期数据执行多因素 Logistic 回归分析。

主要作用：
1. 从 `data1.csv` 读取已经由 `data_clean1.py` 清洗并编码的数据；
2. 按 `data_clean1.py` 中统一后的英文字段名选择自变量与因变量；
3. 绘制自变量相关性热力图并检查多重共线性；
4. 拟合多因素 Logistic 回归模型并输出 OR、P 值和置信区间。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# 统一使用 data_clean1.py 中的清洗后字段名。
FEATURE_COLUMNS = [
    'Sex', 'Age', 'BMI', 'PriorHeadNeckHistory', 'CoronaryHeartDisease', 'Hypertension',
    'PeripheralVascularDisease', 'ImmuneDisease', 'Diabetes', 'Hyperlipidemia', 'SteroidUse',
    'SmokingHistory', 'AlcoholHistory', 'PreopPALB', 'PreopALB', 'PreopHGB',
    'PreopOropharyngealSwab', 'LesionSite', 'ASA', 'PreopAntibiotic', 'OperationDurationMin',
    'IntraopTransfusion', 'AnastomosisType', 'NeckDissection', 'PreopRadiotherapy',
    'PreopChemotherapy', 'PreopConcurrentCRT', 'Endoscopy', 'FlapType', 'Tracheostomy',
    'PostopPathology', 'Differentiation', 'pTNMStage', 'MultiplePrimary', 'PostopDay0to3ALB',
    'PostopDay0to3PALB', 'UnplannedReoperation', 'PulmonaryInfection', 'AnastomoticFistula',
    'DaysToFistulaConfirmation', 'FatLiquefaction', 'MultiDrugResistance'
]
TARGET_COLUMN = 'IncisionInfection'
CORRELATION_PLOT_PATH = 'correlation_heatmap2.png'


# 读取清洗后的建模数据。
try:
    perioperative_data = pd.read_csv('data1.csv')
except FileNotFoundError:
    print("错误：无法找到 'data1.csv' 文件，请先运行 data_clean1.py 生成清洗数据！")
    raise SystemExit(1)


# 按统一字段名提取自变量和因变量。
try:
    X = perioperative_data[FEATURE_COLUMNS].copy()
    y = perioperative_data[TARGET_COLUMN].astype(int)
except KeyError as error:
    print(f"错误：列 {error} 不存在，请检查 data1.csv 是否由最新版本的 data_clean1.py 生成！")
    raise SystemExit(1)


# 清洗后数据理论上已经是数值型；此处保留安全检查，防止旧数据文件混入文本列。
for column in X.columns:
    if X[column].dtype == 'object' or X[column].dtype.name == 'category':
        print(f"\n检测到非数值列 '{column}'，正在进行编码...")
        X[column] = pd.factorize(X[column])[0]
    elif not np.issubdtype(X[column].dtype, np.number):
        print(f"错误：列 '{column}' 包含非数值数据，请检查！")
        raise SystemExit(1)


# 绘制特征相关性热力图，帮助识别线性相关较强的变量。
corr_matrix = X.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    fmt='.2f',
    square=True,
)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CORRELATION_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()


# 若变量间绝对相关系数超过阈值，则删除其中一列以降低多重共线性影响。
print("\n检查多重共线性（相关系数 > 0.60）...")
corr_matrix = X.corr(numeric_only=True).abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.6)]
if columns_to_drop:
    print(f"警告：以下列存在高相关性，将移除：{columns_to_drop}")
    X = X.drop(columns=columns_to_drop)


# 添加截距项并拟合多因素 Logistic 回归模型。
X = sm.add_constant(X)
print("\n=== 多因素 Logistic 回归分析 ===")
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=1, maxiter=200)
print("\n模型概要：")
print(result.summary())


# 汇总每个自变量的回归系数、OR 值、P 值和 95% 置信区间。
results = []
confidence_intervals = result.conf_int()
for column in X.columns:
    if column == 'const':
        continue

    coefficient = result.params[column]
    confidence_interval = confidence_intervals.loc[column]
    p_value = result.pvalues[column]
    odds_ratio = np.exp(coefficient)
    results.append({
        '变量': column,
        '回归系数': round(coefficient, 4),
        '优势比 (OR)': round(odds_ratio, 4),
        'P值': round(p_value, 4),
        '95% 置信区间下限 (OR)': round(np.exp(confidence_interval[0]), 4),
        '95% 置信区间上限 (OR)': round(np.exp(confidence_interval[1]), 4),
    })

results_df = pd.DataFrame(results)
print("\n多因素 Logistic 回归结果：")
print(results_df)


# 按阈值筛选潜在独立危险因素，便于后续进一步解释。
independent_factors = results_df[results_df['P值'] < 0.1]
print("\n=== 独立危险因素 (P < 0.1) ===")
print(independent_factors)
