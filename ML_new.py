import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import scipy.stats as stats

# ========== 文件夹准备 ==========
output_root = "analysis_results"
if not os.path.exists(output_root):
    os.makedirs(output_root)


# ========== 辅助函数：DeLong 检验 ==========
def delong_roc_test(y_true, prob_a, prob_b):
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)
    delta_auc = auc_a - auc_b
    var_a = (auc_a * (1 - auc_a)) / len(y_true)
    var_b = (auc_b * (1 - auc_b)) / len(y_true)
    z = delta_auc / np.sqrt(var_a + var_b + 1e-8)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return delta_auc, p_value


# ========== 辅助函数：Bootstrap 验证 ==========
def bootstrap_auc(y_true, y_prob, n_iterations=1000):
    stats_list = []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    for i in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        stats_list.append(score)
    if not stats_list: return 0.0, 0.0
    return np.percentile(stats_list, [2.5, 97.5])


# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载与准备数据 ==========
data = pd.read_csv("ca1_cleaned2.csv")


# 定义两个实验的特征集
exp_features = {
    "Exp1": ['Sex', 'ASA','Multi-drug resistance'],
    "Exp2": ['Multi-drug resistance', 'Duration of surgery', 'Post-operative admission to ICU', 'Sex', 'ASA','Postoperative 0-3 days CRP']
}

all_exp_results = {}
final_probs_for_delong = {}

# ========== 开始两个实验的循环 ==========
for exp_name, features in exp_features.items():
    print(f"\n{'=' * 20} 正在执行：{exp_name} {'=' * 20}")

    exp_folder = os.path.join(output_root, exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    X = data[features]
    y = data['Infection status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        'GLM': LogisticRegression(class_weight='balanced', solver='liblinear'),
        'RF': RandomForestClassifier(class_weight='balanced', random_state=42),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
        'NNET': MLPClassifier(max_iter=1000, random_state=42),
        'GBM': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }

    param_grids = {
        'GLM': {'classifier__C': [0.1, 1, 10]},
        'RF': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
        'SVM': {'classifier__C': [0.1, 1], 'classifier__kernel': ['linear', 'rbf']},
        'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__alpha': [0.0001, 0.001]},
        'GBM': {'classifier__max_depth': [3, 5], 'classifier__learning_rate': [0.01, 0.1],
                'classifier__n_estimators': [50, 100]}
    }

    results = {}
    exp_model_probs = {}

    for name, model in models.items():
        print(f"训练模型：{name}")
        pipeline = ImbPipeline(steps=[
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        print(f"  {name} 最佳参数: {grid.best_params_}")

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        exp_model_probs[name] = y_prob

        auc = roc_auc_score(y_test, y_prob)
        ci_low, ci_high = bootstrap_auc(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            'AUC': auc,
            '95% CI_Lower': ci_low,
            '95% CI_Upper': ci_high,
            '准确性': accuracy_score(y_test, y_pred),
            '敏感性': tp / (tp + fn) if (tp + fn) > 0 else 0,
            '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1': f1_score(y_test, y_pred)
        }

        if name == 'GLM':
            final_probs_for_delong[exp_name] = y_prob

        # --- 保存 Permutation Importance 图 ---
        perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        plt.figure(figsize=(8, 5))
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx], color='skyblue')
        plt.title(f"{exp_name} - {name} Permutation Importance")
        plt.xlabel("Accuracy Decrease")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_folder, f"{name}_permutation_importance.png"), dpi=300)
        plt.close()

        # --- 保存 SHAP 图 (针对 GBM 和 RF) ---
        if name in ['GBM', 'RF']:
            try:
                classifier = best_model.named_steps['classifier']
                X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
                # 使用通用 Explainer
                explainer = shap.Explainer(classifier, X_test_scaled)
                shap_values = explainer(X_test_scaled)

                plt.figure()
                # 如果是随机森林，shap_values 可能会包含两个类别的维度，取正类 [..., 1]
                if len(shap_values.shape) == 3:
                    shap.summary_plot(shap_values[:, :, 1], X_test_scaled, feature_names=features, show=False)
                else:
                    shap.summary_plot(shap_values, X_test_scaled, feature_names=features, show=False)

                plt.title(f"{exp_name} - {name} SHAP Summary")
                plt.savefig(os.path.join(exp_folder, f"{name}_shap_summary.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"      {name} SHAP 绘图跳过: {e}")

    # --- 保存该实验的 ROC 曲线 ---
    plt.figure(figsize=(8, 6))
    for m_name, m_prob in exp_model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, m_prob)
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={results[m_name]['AUC']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"{exp_name} ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(exp_folder, "overall_roc_curves.png"), dpi=300)
    plt.close()

    # 将结果转为 DataFrame 并保存为该实验的独立 CSV
    df_res = pd.DataFrame(results).T
    df_res.to_csv(os.path.join(output_root, f"{exp_name}_metrics.csv"), encoding='utf-8-sig')
    all_exp_results[exp_name] = df_res

# ========== 3. ΔAUC 与 DeLong 检验 ==========
if 'Exp1' in final_probs_for_delong and 'Exp2' in final_probs_for_delong:
    delta_auc, p_val = delong_roc_test(y_test.values, final_probs_for_delong['Exp2'], final_probs_for_delong['Exp1'])

    with open(os.path.join(output_root, "delong_test_result.txt"), "w", encoding='utf-8') as f:
        f.write(f"实验对比总结 (Logistic Model)\n")
        f.write(f"实验一 AUC: {roc_auc_score(y_test, final_probs_for_delong['Exp1']):.4f}\n")
        f.write(f"实验二 AUC: {roc_auc_score(y_test, final_probs_for_delong['Exp2']):.4f}\n")
        f.write(f"ΔAUC (Exp2 - Exp1): {delta_auc:.4f}\n")
        f.write(f"DeLong 检验 P-value: {p_val:.4f}\n")

plt.figure(figsize=(8, 7))

for exp_name in ['Exp1', 'Exp2']:
    # 我们以每个实验中的 Logistic 或最优模型为例
    y_prob = final_probs_for_delong[exp_name]

    # 计算 PR 曲线数据
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = sklearn_auc(recall, precision)

    plt.plot(recall, precision, lw=2, label=f'{exp_name} (AUPRC = {auprc:.3f})')

plt.xlabel('Recall (敏感性/不漏诊率)')
plt.ylabel('Precision (精确率/预测准确率)')
plt.title('PR 曲线对比：早期模型 vs 围术期模型')
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# 保存 PR 曲线
plt.savefig(os.path.join(output_root, "PR_Curve_Comparison.png"), dpi=300)
plt.show()
print(f"\n任务完成！指标已保存为 CSV，图片已保存至文件夹: {output_root}")