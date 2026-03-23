# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260323_201130_BJT`
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
- THRESHOLD_CV_SPLITS: 5
- BOOTSTRAP_ROUNDS: 1000
- THRESHOLD_GRID_SIZE: 201
- MIN_SPECIFICITY_FLOOR: 0.6
- N_JOBS: 1

## 输出文件
- 指标汇总: `ml_model_metrics.csv`
- Bootstrap汇总: `bootstrap_auc_summary.csv`
- 最佳参数: `ml_model_best_params.json`
- ROC图: `roc_curves.png`
- SHAP目录: `shap`

## 模型结果预览
| model   |   selected_threshold |   threshold_train_sensitivity |   threshold_train_specificity |   tuning_auc |   cv_auc_mean |   cv_auc_std |   test_auc |   accuracy |   sensitivity |   specificity |        f1 |   bootstrap_auc_ci_lower |   bootstrap_auc_ci_median |   bootstrap_auc_ci_upper |   n_features |
|:--------|---------------------:|------------------------------:|------------------------------:|-------------:|--------------:|-------------:|-----------:|-----------:|--------------:|--------------:|----------:|-------------------------:|--------------------------:|-------------------------:|-------------:|
| NNET    |            0.0166424 |                      0.8      |                      0.632723 |     0.823899 |      0.81762  |     0.123094 |   0.743093 |   0.795276 |      0.461538 |      0.801068 | 0.0714286 |                 0.636227 |                  0.730204 |                 0.743299 |            6 |
| GLM     |            0.421105  |                      0.833333 |                      0.600686 |     0.809978 |      0.820996 |     0.119247 |   0.735083 |   0.557743 |      0.769231 |      0.554072 | 0.0560224 |                 0.589951 |                  0.710255 |                 0.7413   |            6 |
| SVM     |            0.013293  |                      0.9      |                      0.605263 |     0.825988 |      0.826635 |     0.118205 |   0.734158 |   0.551181 |      0.769231 |      0.547397 | 0.0552486 |                 0.645417 |                  0.732618 |                 0.739504 |            6 |
| GBM     |            0.0201511 |                      0.733333 |                      0.776316 |     0.798377 |      0.806659 |     0.120453 |   0.732618 |   0.793963 |      0.461538 |      0.799733 | 0.0710059 |                 0.553977 |                  0.706737 |                 0.739962 |            6 |
| XGBoost |            0.0168131 |                      0.8      |                      0.655606 |     0.763865 |      0.789105 |     0.117281 |   0.680086 |   0.784777 |      0.384615 |      0.791722 | 0.0574713 |                 0.563329 |                  0.618748 |                 0.72841  |            6 |
| RF      |            0.424374  |                      0.8      |                      0.610412 |     0.794346 |      0.800052 |     0.122109 |   0.657235 |   0.562992 |      0.615385 |      0.562083 | 0.0458453 |                 0.596588 |                  0.652768 |                 0.698436 |            6 |

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
