# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260323_104431_UTC`
- 训练集路径: `analysis_outputs\train_set.csv`
- 测试集路径: `analysis_outputs\test_set.csv`
- 特征数量: 6
- 特征列表: PreopConcurrentCRT, NeckDissection, Pathogen, MultiDrugResistance, AnastomoticFistula, AlcoholHistory

## 关键参数
- RANDOM_STATE: 42
- CV_SPLITS: 10
- CV_REPEATS: 100
- TUNING_CV_SPLITS: 5
- TUNING_CV_REPEATS: 5
- TUNING_ITER: 20
- BOOTSTRAP_ROUNDS: 1000
- N_JOBS: 1

## 输出文件
- 指标汇总: `ml_model_metrics.csv`
- Bootstrap汇总: `bootstrap_auc_summary.csv`
- 最佳参数: `ml_model_best_params.json`
- ROC图: `roc_curves.png`
- SHAP目录: `shap`

## 模型结果预览
| model   |   tuning_auc |   cv_auc_mean |   cv_auc_std |   test_auc |   accuracy |   sensitivity |   specificity |        f1 |   bootstrap_auc_ci_lower |   bootstrap_auc_ci_median |   bootstrap_auc_ci_upper |   n_features |
|:--------|-------------:|--------------:|-------------:|-----------:|-----------:|--------------:|--------------:|----------:|-------------------------:|--------------------------:|-------------------------:|-------------:|
| NNET    |     0.823899 |      0.81762  |     0.123094 |   0.743093 |   0.98294  |      0        |      1        | 0         |                 0.636227 |                  0.730204 |                 0.743299 |            6 |
| GLM     |     0.809978 |      0.820996 |     0.119247 |   0.735083 |   0.787402 |      0.461538 |      0.793057 | 0.0689655 |                 0.589951 |                  0.710255 |                 0.7413   |            6 |
| SVM     |     0.825988 |      0.826635 |     0.118205 |   0.734158 |   0.981627 |      0        |      0.998665 | 0         |                 0.645417 |                  0.732618 |                 0.739504 |            6 |
| GBM     |     0.798377 |      0.806659 |     0.120453 |   0.732618 |   0.980315 |      0        |      0.99733  | 0         |                 0.553977 |                  0.706737 |                 0.739962 |            6 |
| XGBoost |     0.763865 |      0.789105 |     0.117281 |   0.680086 |   0.981627 |      0        |      0.998665 | 0         |                 0.563329 |                  0.618748 |                 0.72841  |            6 |
| RF      |     0.794346 |      0.800052 |     0.122109 |   0.657235 |   0.791339 |      0.230769 |      0.801068 | 0.0363636 |                 0.596588 |                  0.652768 |                 0.698436 |            6 |

## 各模型最佳参数
### GLM
- best_cv_auc: 0.809978
- best_params: `{"classifier__C": 0.01, "classifier__penalty": "l2"}`

### RF
- best_cv_auc: 0.794346
- best_params: `{"class_weight": "balanced", "max_depth": 5, "max_features": "log2", "min_samples_leaf": 10, "min_samples_split": 2, "n_estimators": 800}`

### SVM
- best_cv_auc: 0.825988
- best_params: `{"classifier__C": 0.1, "classifier__gamma": 0.001, "classifier__kernel": "rbf"}`

### NNET
- best_cv_auc: 0.823899
- best_params: `{"classifier__activation": "tanh", "classifier__alpha": 0.01, "classifier__batch_size": 16, "classifier__hidden_layer_sizes": [32], "classifier__learning_rate_init": 0.01, "classifier__solver": "adam"}`

### GBM
- best_cv_auc: 0.798377
- best_params: `{"learning_rate": 0.01, "max_depth": 3, "min_samples_leaf": 1, "n_estimators": 300, "subsample": 0.8}`

### XGBoost
- best_cv_auc: 0.763865
- best_params: `{"colsample_bytree": 0.8, "gamma": 0.3, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 500, "reg_alpha": 0.1, "reg_lambda": 2.0, "subsample": 0.8}`
