"""基于清洗后的围手术期数据训练并评估多种机器学习模型。

主要作用：
1. 从 `data1.csv` 读取 `data_clean1.py` 产出的标准化数据；
2. 使用与清洗脚本一致的字段名构建特征和标签；
3. 训练 GLM、随机森林、SVM、神经网络、朴素贝叶斯和 XGBoost 模型；
4. 输出模型性能、ROC/DCA 曲线以及 SHAP 解释结果。
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore', category=ConvergenceWarning)


# =================== 数据 ===================
# 与 data_clean1.py 对齐：这里所有字段名均使用清洗后的标准英文名。
DATA_PATH = 'data1.csv'
FEATURE_COLUMNS = [
    'PreopPALB', 'PreopALB', 'PreopHGB', 'PreopOropharyngealSwab',
    'PreopAntibiotic', 'FlapType',
    'PostopPathology', 'pTNMStage', 'MultiplePrimary',
    'AnastomoticFistula'
]
TARGET_COLUMN = 'IncisionInfection'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f'未找到 {DATA_PATH}')

perioperative_data = pd.read_csv(DATA_PATH)
X = perioperative_data[FEATURE_COLUMNS].copy()
y = perioperative_data[TARGET_COLUMN].astype(int)

# FlapType 属于类别字段；即使清洗结果已编码，这里仍兼容字符串输入场景。
X['FlapType'] = LabelEncoder().fit_transform(X['FlapType'].astype(str))

# 划分训练集与测试集，用于后续模型训练与泛化评估。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


# =================== 时间戳文件夹 ===================
# 使用当前时间创建输出目录，避免覆盖历史结果。
time_tag = datetime.now().strftime('%H-%M')
FIG_DIR = f'results_figures_{time_tag}'
os.makedirs(FIG_DIR, exist_ok=True)


# =================== 工具函数 ===================
def calculate_metrics(y_true, y_prob, threshold=0.5):
    """根据预测概率计算 AUC、敏感性、特异性、准确率和 F1。"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (
        roc_auc_score(y_true, y_prob),
        tp / (tp + fn) if (tp + fn) else 0,
        tn / (tn + fp) if (tn + fp) else 0,
        (tp + tn) / (tp + tn + fp + fn),
        f1_score(y_true, y_pred),
    )


def calculate_net_benefit(y_true, y_prob, thresholds):
    """按一组阈值计算 DCA 所需的净受益。"""
    y_true = np.array(y_true)
    net_benefits = []
    sample_size = len(y_true)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit = (
            (true_positive / sample_size) - (false_positive / sample_size) * (threshold / (1 - threshold))
            if 0 < threshold < 1 else 0
        )
        net_benefits.append(net_benefit)
    return net_benefits


def save_shap_summary(shap_values, X_data, model_name):
    """保存 SHAP summary 图，用于展示特征对预测结果的总体影响。"""
    plt.figure()
    shap.summary_plot(shap_values, X_data, show=False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/SHAP_{model_name}_summary.png', dpi=300)
    plt.close()


def save_shap_bar(shap_values, X_data, model_name):
    """保存 SHAP 条形图，用于展示特征重要性排序。"""
    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/SHAP_{model_name}_bar.png', dpi=300)
    plt.close()


# =================== 中文显示 ===================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =================== 模型 ===================
# 为不同模型定义基础估计器，统一放入带 SMOTE 的管道中训练。
models = {
    'GLM': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', dual=False, max_iter=10000),
        cv=5,
    ),
    'NNET': MLPClassifier(max_iter=500, random_state=42),
    'NB': GaussianNB(),
}

param_grids = {
    'GLM': {'classifier__C': [1]},
    'RF': {'classifier__n_estimators': [100], 'classifier__max_depth': [20]},
    'SVM': {'classifier__estimator__C': [0.1, 1]},
    'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)]},
    'NB': {},
}

results = {}
model_probs = {}
CV = 5


# =================== 训练 + SHAP ===================
# 逐个训练模型，记录性能指标，并尝试生成 SHAP 可解释性图。
for model_name, model in models.items():
    print(f'\n>>> 训练模型: {model_name}')

    training_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', model),
    ])

    grid_search = GridSearchCV(training_pipeline, param_grids[model_name], cv=CV, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_prob = best_model.predict_proba(X_test)[:, 1]
    model_probs[model_name] = y_prob

    auc, sensitivity, specificity, accuracy, f1 = calculate_metrics(y_test, y_prob)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)

    results[model_name] = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1,
        'fpr': false_positive_rate,
        'tpr': true_positive_rate,
    }

    classifier = best_model.named_steps['classifier']
    background_data = X_train.sample(min(100, len(X_train)), random_state=42)
    explanation_data = X_test.sample(min(50, len(X_test)), random_state=42)

    try:
        if model_name == 'RF':
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(explanation_data)[1]
        elif model_name == 'GLM':
            explainer = shap.LinearExplainer(classifier, background_data)
            shap_values = explainer.shap_values(explanation_data)
        else:
            explainer = shap.KernelExplainer(
                lambda input_data: classifier.predict_proba(pd.DataFrame(input_data, columns=X.columns))[:, 1],
                background_data,
            )
            shap_values = explainer.shap_values(explanation_data, nsamples=100)

        save_shap_summary(shap_values, explanation_data, model_name)
        save_shap_bar(shap_values, explanation_data, model_name)

    except Exception as error:
        print(f'{model_name} SHAP 失败: {error}')


# =================== XGBoost ===================
# 单独训练 XGBoost，并与其他模型一起纳入统一评估与可解释性输出。
print('\n>>> 训练模型: XGBoost')
negative_count, positive_count = np.bincount(y_train)
scale_pos_weight = negative_count / positive_count

xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

grid_search = GridSearchCV(
    xgb_model,
    {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1]},
    cv=CV,
    scoring='roc_auc',
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

classifier = grid_search.best_estimator_
y_prob = classifier.predict_proba(X_test)[:, 1]
model_probs['GBM'] = y_prob

auc, sensitivity, specificity, accuracy, f1 = calculate_metrics(y_test, y_prob)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)

results['GBM'] = {
    'AUC': auc,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Accuracy': accuracy,
    'F1': f1,
    'fpr': false_positive_rate,
    'tpr': true_positive_rate,
}

explainer = shap.TreeExplainer(classifier)
shap_data = X_test.sample(min(50, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_data)
save_shap_summary(shap_values, shap_data, 'GBM')
save_shap_bar(shap_values, shap_data, 'GBM')


# =================== 指标表 ===================
# 输出各模型核心性能指标，便于对比。
metrics_df = pd.DataFrame(results).T[['AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'F1']]
metrics_df.to_csv(f'{FIG_DIR}/model_metrics.csv', encoding='utf-8-sig')
print('\n模型性能指标：\n', metrics_df.round(4))


# =================== ROC ===================
# 使用双重平滑绘制圆滑版 ROC 曲线，便于汇报展示。
plt.figure(figsize=(8, 6))
mean_fpr = np.linspace(0, 1, 3000)

for model_name, model_result in results.items():
    tpr_interp = np.interp(mean_fpr, model_result['fpr'], model_result['tpr'])

    window = 81
    kernel = np.ones(window) / window
    tpr_smooth = np.convolve(tpr_interp, kernel, mode='same')
    tpr_smooth = np.convolve(tpr_smooth, kernel, mode='same')

    tpr_smooth = np.maximum.accumulate(tpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)

    plt.plot(mean_fpr, tpr_smooth, label=f'{model_name} ')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlabel('1 - 特异性')
plt.ylabel('敏感性')
plt.title('ROC 曲线')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/ROC_curves_extra_round.png', dpi=300)
plt.close()


# =================== DCA ===================
# 计算并保存 DCA 曲线，用于比较不同模型的临床净受益。
thresholds = np.linspace(0.05, 0.6, 50)
plt.figure(figsize=(8, 6))
for model_name, probabilities in model_probs.items():
    plt.plot(thresholds, calculate_net_benefit(y_test, probabilities, thresholds), label=model_name)
plt.xlabel('阈值概率')
plt.ylabel('净受益')
plt.title('DCA 曲线')
plt.legend()
plt.savefig(f'{FIG_DIR}/DCA_curves.png', dpi=300)
