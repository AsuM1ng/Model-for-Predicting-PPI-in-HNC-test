import pandas as pd
import statsmodels.api as sm
import numpy as np

# 加载数据
try:
    data = pd.read_csv(r"data1.csv")
except FileNotFoundError:
    print("错误：无法找到 'data1.csv' 文件，请检查文件路径！")
    exit()

# 分离自变量和因变量
try:

    selected_columns = ['gender', 'age', 'BMI', 'history_head_neck_surgery', 'coronary heart_disease', 'hypertension',
                   'peripheralvascular_disease', 'immune_disease', 'diabetes', 'hyperlipidemia', 'hormone_use',
                   'smoking_history', 'alcohol_history', 'pre_op_palb', 'pre_op_alb', 'pre_op_hgb',
                   'pre_op_oropharyngeal_swab', 'lesion_site', 'asa_score', 'pre_op_antibiotics', 'surgery_duration',
                   'intraop_transfusion', 'anastomosis_type', 'neck_dissection', 'pre_op_radiotherapy',
                   'pre_op_chemotherapy', 'pre_op_chemoradiotherapy', 'endoscopic_surgery', 'flap_type', 'tracheostomy',
                   'post_op_pathology', 'differentiation', 'ptnm_stage', 'multiple_primary', 'post_op_min_alb',
                   'post_op_palb', 'unplanned_reoperation', 'pulmonary_infection', 'anastomotic_leak',
                   'leak_confirm_day', 'fat_liquefaction', 'multidrug_resistant']
    #新的
    X = data[selected_columns]
    y = data['wound_infection']
except KeyError as e:
    print(f"错误：列 {e} 不存在，请检查列名！")
    exit()

# 确保 y 是二分类变量（0 和 1）
if y.dtype not in [np.int32, np.int64, np.float32, np.float64]:
    print("\n因变量 'pulmonary_infection' 非数值型，正在处理...")
    if y.nunique() == 2:
        y = pd.factorize(y)[0]
        print(f"因变量已转换为：{np.unique(y)}")
    else:
        print(f"错误：因变量 'pulmonary_infection' 有 {y.nunique()} 个唯一值，不是二分类！")
        exit()

# 处理 X 的非数值列
for column in X.columns:
    if X[column].dtype == 'object' or X[column].dtype.name == 'category':
        print(f"\n检测到非数值列 '{column}'，正在进行编码...")
        X[column] = pd.factorize(X[column])[0]
    elif not np.issubdtype(X[column].dtype, np.number):
        print(f"错误：列 '{column}' 包含非数值数据，请检查！")
        exit()

# 检查并处理缺失值
if X.isnull().values.any():
    print("存在缺失值 (NaN)，正在填充...")
    X = X.fillna(X.mean())  # 用均值填充缺失值

if np.isinf(X.values).any():
    print("存在无穷大值 (Inf)，正在处理...")
    X = X.replace([np.inf, -np.inf], np.nan)  # 将 Inf 替换为 NaN
    X = X.dropna()  # 删除含有 NaN 的行

# 添加常数项
X = sm.add_constant(X)

# 单因素 Logistic 回归
results = []

for column in X.columns:
    if column != 'const':  # 避免对常数项进行分析
        model = sm.Logit(y, X[[column, 'const']])  # 单因素回归
        result = model.fit(disp=0)  # disp=0 不显示迭代过程
        coef = result.params[column]
        conf_int = result.conf_int().loc[column]
        p_value = result.pvalues[column]
        odds_ratio = np.exp(coef)

        results.append({
            '变量': column,
            '回归系数': round(coef, 4),
            '优势比 (OR)': round(odds_ratio, 4),
            'P值': round(p_value, 4),
            '95% 置信区间下限 (OR)': round(np.exp(conf_int[0]), 4),
            '95% 置信区间上限 (OR)': round(np.exp(conf_int[1]), 4)
        })

# 输出结果
results_df = pd.DataFrame(results)
print("\n单因素 Logistic 回归结果：")
print(results_df)

# 筛选独立危险因素（P < 0.05）
independent_factors = results_df[results_df['P值'] < 0.05]
print("\n=== 独立危险因素 (P < 0.05) ===")
print(independent_factors)
