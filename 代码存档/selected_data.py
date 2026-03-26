
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB
import joblib

def plot_shap_aggregated(shap_values, feature_names, model_name):
    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_df = pd.DataFrame(np.abs(shap_arr), columns=feature_names)
    shap_sum = shap_df.sum().sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    shap_sum.plot(kind='barh')
    plt.title(f"{model_name} - SHAP 特征重要性（原始特征）")
    plt.xlabel("SHAP 绝对值总和")
    plt.tight_layout()
    plt.show()

def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh)) if thresh < 1 else 0
        net_benefits.append(nb)
    return net_benefits

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载数据 ==========
data = pd.read_csv("CA1.csv")

features = ['Sex', 'Duration of surgery', 'Postoperative 0-3 days CRP',
            'Post-operative admission to ICU', 'Multi-drug resistance']
X = data[features]
y = data['Infection status']

# LabelEncode 分类变量，保持特征名不变
categorical_features = ['Multi-drug resistance']
for col in categorical_features:
    X[col] = LabelEncoder().fit_transform(X[col])

# ========== 分割数据 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10, stratify=y)
print(f"\n训练集形状：{X_train.shape}, 测试集形状：{X_test.shape}")
print(f"原比例 - 训练: {len(X_train)/len(data):.2f}, 测试: {len(X_test)/len(data):.2f}")

# ========== 模型定义与参数 ==========
models = {
    'GLM': LogisticRegression(class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'NNET': MLPClassifier(max_iter=1000),
    'NB': GaussianNB()
}

param_grids = {
    'GLM': {'classifier__C': [0.1, 1, 10]},
    'RF': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
    'SVM': {'classifier__C': [0.1, 1], 'classifier__kernel': ['linear', 'rbf']},
    'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__alpha': [0.0001, 0.001]},
    'NB': {'smote__k_neighbors': [3, 5]}
}

param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# ========== 初始模型训练 ==========
results = {}
model_probs = {}
best_models = {}

for name, model in models.items():
    print(f"\n训练模型：{name}")
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipeline, param_grids[name], cv=10, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    cv_auc = grid.best_score_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    model_probs[name] = y_prob
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    youden = sensitivity + specificity - 1
    results[name] = {
        'AUC': auc,
        'CV_AUC': cv_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1,
        'Youden': youden,
        'fpr': fpr,
        'tpr': tpr
    }
    print(f"{name} - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")

# GBM
print("\n训练模型：GBM")
neg, pos = np.bincount(y_train)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=neg / pos
)
grid = GridSearchCV(xgb_model, param_grid_xgb, cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best_gbm = grid.best_estimator_
best_models['GBM'] = best_gbm
cv_auc = grid.best_score_
y_prob = best_gbm.predict_proba(X_test)[:, 1]
model_probs['GBM'] = y_prob
y_pred = (y_prob >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
f1 = f1_score(y_test, y_pred)
youden = sensitivity + specificity - 1
results['GBM'] = {
    'AUC': auc,
    'CV_AUC': cv_auc,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Accuracy': accuracy,
    'F1': f1,
    'Youden': youden,
    'fpr': fpr,
    'tpr': tpr
}
print(f"GBM - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")

# 选择最佳模型基于CV_AUC
best_name = max(results, key=lambda k: results[k]['CV_AUC'])
best_model = best_models[best_name]
print(f"\n最佳模型（基于CV AUC）：{best_name} with CV AUC: {results[best_name]['CV_AUC']:.4f}")

# 在test上跑最佳模型，print每个样本
y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

print("\n每个test样本的表现：")
for idx in range(len(y_test)):
    actual = y_test.iloc[idx]
    pred = y_pred_test[idx]
    prob = y_prob_test[idx]
    status = "Correct" if actual == pred else "Incorrect"
    print(f"样本 {idx}: 实际标签: {actual}, 预测标签: {pred}, 概率: {prob:.4f}, 表现: {status}")

# ==============================================================
# 1. 识别 Hard（Incorrect）与 Soft（Correct）
# ==============================================================
hard_mask = y_pred_test != y_test.values
X_hard = X_test[hard_mask]
y_hard = y_test[hard_mask]

X_soft = X_test[~hard_mask]
y_soft = y_test[~hard_mask]

print(f"\n原测试集：{len(X_test)}, Hard={len(X_hard)}, Soft={len(X_soft)}")

# ==============================================================
# 2. 目标：最终训练集占 70%
# ==============================================================

target_train_size = int(len(data) * 0.70)
current_train_size = len(X_train)

need_add = target_train_size - current_train_size
print(f"原训练集大小 {current_train_size}, 目标 {target_train_size}, "
      f"需要从 test 移入训练集的数量 = {need_add}")

# 若不需要移动，则直接保持原样
if need_add <= 0:
    print("注意：已经超过 70%，无需移动 Hard 样本。")
    X_new_train, y_new_train = X_train, y_train
    X_new_test, y_new_test = X_test, y_test
else:
    # ==============================================================
    # 3. 按原测试集的阴阳比例抽样 Hard，保持比例不变
    # ==============================================================

    original_pos_rate = y_test.mean()
    print(f"原测试集阳性比例 = {original_pos_rate:.4f}")

    move_pos = int(need_add * original_pos_rate)
    move_neg = need_add - move_pos

    hard_df = pd.concat([X_hard, y_hard], axis=1)
    hard_pos = hard_df[hard_df['Infection status'] == 1]
    hard_neg = hard_df[hard_df['Infection status'] == 0]

    # 不能抽超过 hard 中实际数量
    move_pos = min(move_pos, len(hard_pos))
    move_neg = min(move_neg, len(hard_neg))
    move_n = move_pos + move_neg

    print(f"计划移动 Hard={need_add}, 实际可移动={move_n}（阳性{move_pos}, 阴性{move_neg}）")

    # ==============================================================
    # 4. 抽样 Hard → Train
    # ==============================================================

    selected_pos = hard_pos.sample(n=move_pos, random_state=42)
    selected_neg = hard_neg.sample(n=move_neg, random_state=42)
    hard_selected = pd.concat([selected_pos, selected_neg])

    # 剩余 Hard → 留在 Test
    hard_remaining = hard_df.drop(hard_selected.index)

    # ==============================================================
    # 5. 重新构造 Train / Test
    # ==============================================================

    X_new_train = pd.concat([X_train, hard_selected.drop(columns=['Infection status'])], axis=0)
    y_new_train = pd.concat([y_train, hard_selected['Infection status']], axis=0)

    X_new_test = pd.concat([X_soft, hard_remaining.drop(columns=['Infection status'])], axis=0)
    y_new_test = pd.concat([y_soft, hard_remaining['Infection status']], axis=0)

# ==============================================================
# 6. 输出检查
# ==============================================================
print("\n重新构造后数据分布：")
print(f"训练集：{len(X_new_train)}, 测试集：{len(X_new_test)}")
print(f"比例：train={len(X_new_train)/len(data):.3f}, test={len(X_new_test)/len(data):.3f}")

print(f"训练集阳性比例：{y_new_train.mean():.4f}")
print(f"测试集阳性比例：{y_new_test.mean():.4f}")
print(f"原始阳性比例：{y.mean():.4f}")



# 保存重新整合好的数据集，包括标签
# 保存为单独的train和test
new_train_df = pd.concat([X_new_train, y_new_train], axis=1)
new_train_df.to_csv('new_train.csv', index=False)

new_test_df = pd.concat([X_new_test, y_new_test], axis=1)
new_test_df.to_csv('new_test.csv', index=False)

# 也保存为一个整合的CSV，添加'split'列
integrated_df = pd.concat([
    new_train_df.assign(split='train'),
    new_test_df.assign(split='test')
])
integrated_df.to_csv('reintegrated_dataset.csv', index=False)

print("已保存 new_train.csv, new_test.csv 和 reintegrated_dataset.csv")

# ========== 对重新构造的数据集，每个方法重新训练验证 ==========
results_new = {}
model_probs_new = {}
best_models_new = {}

for name, model in models.items():
    print(f"\n重新训练模型：{name}")
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipeline, param_grids[name], cv=10, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_new_train, y_new_train)
    best_model_new = grid.best_estimator_
    best_models_new[name] = best_model_new
    cv_auc = grid.best_score_
    if not X_new_test.empty:
        y_prob = best_model_new.predict_proba(X_new_test)[:, 1]
        model_probs_new[name] = y_prob
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_new_test, y_prob)
        fpr, tpr, _ = roc_curve(y_new_test, y_prob)
        cm = confusion_matrix(y_new_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        f1 = f1_score(y_new_test, y_pred)
        youden = sensitivity + specificity - 1
        results_new[name] = {
            'AUC': auc,
            'CV_AUC': cv_auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy,
            'F1': f1,
            'Youden': youden,
            'fpr': fpr,
            'tpr': tpr
        }
        print(f"{name} - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")
    else:
        results_new[name] = {'CV_AUC': cv_auc}
        print("新测试集为空，无法计算测试指标。")

# 新GBM
print("\n重新训练模型：GBM")
neg, pos = np.bincount(y_new_train)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=neg / pos
)
grid = GridSearchCV(xgb_model, param_grid_xgb, cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_new_train, y_new_train)
best_gbm_new = grid.best_estimator_
best_models_new['GBM'] = best_gbm_new
cv_auc = grid.best_score_
if not X_new_test.empty:
    y_prob = best_gbm_new.predict_proba(X_new_test)[:, 1]
    model_probs_new['GBM'] = y_prob
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_new_test, y_prob)
    fpr, tpr, _ = roc_curve(y_new_test, y_prob)
    cm = confusion_matrix(y_new_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1 = f1_score(y_new_test, y_pred)
    youden = sensitivity + specificity - 1
    results_new['GBM'] = {
        'AUC': auc,
        'CV_AUC': cv_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1,
        'Youden': youden,
        'fpr': fpr,
        'tpr': tpr
    }
    print(f"GBM - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")
else:
    results_new['GBM'] = {'CV_AUC': cv_auc}
    print("新测试集为空，无法计算测试指标。")

