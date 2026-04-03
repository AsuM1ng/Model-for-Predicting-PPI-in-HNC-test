# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260325_150139_BJT`
- 训练集路径: `analysis_outputs\train_set.csv`
- 测试集路径: `analysis_outputs\test_set.csv`
- 特征数量: 5
- 特征列表: OperationDurationMin, PreopConcurrentCRT, NeckDissection, IntraopTransfusion, Tracheostomy

## 关键参数
- RANDOM_STATE: 42
- CV_SPLITS: 10
- CV_REPEATS: 10
- TUNING_CV_SPLITS: 5
- TUNING_CV_REPEATS: 5
- TUNING_ITER: 20
- THRESHOLD_CV_SPLITS: 5
- BOOTSTRAP_ROUNDS: 100
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
| XGBoost |            0.0146753 |                      0.827586 |                      0.622608 |     0.774361 |      0.767741 |     0.157531 |   0.884414 |   0.569273 |      0.916667 |      0.563459 | 0.0654762 |                 0.734571 |                  0.83374  |                 0.89512  |            5 |
| SVM     |            0.0137095 |                      0.724138 |                      0.653708 |     0.799299 |      0.79809  |     0.135183 |   0.845479 |   0.813443 |      0.666667 |      0.8159   | 0.105263  |                 0.171414 |                  0.844142 |                 0.861285 |            5 |
| GBM     |            0.0119716 |                      0.689655 |                      0.626196 |     0.741981 |      0.745295 |     0.162179 |   0.839435 |   0.565158 |      0.916667 |      0.559275 | 0.0648968 |                 0.665451 |                  0.814011 |                 0.860298 |            5 |
| GLM     |            0.429557  |                      0.793103 |                      0.60945  |     0.791818 |      0.790961 |     0.144867 |   0.83897  |   0.580247 |      0.916667 |      0.574616 | 0.0670732 |                 0.768619 |                  0.837517 |                 0.875558 |            5 |
| NNET    |            0.0117038 |                      0.793103 |                      0.614833 |     0.787766 |      0.789469 |     0.144789 |   0.832345 |   0.644719 |      0.916667 |      0.640167 | 0.0782918 |                 0.718561 |                  0.857218 |                 0.885635 |            5 |
| RF      |            0.340332  |                      0.793103 |                      0.621411 |     0.764722 |      0.758202 |     0.153849 |   0.816423 |   0.607682 |      0.916667 |      0.60251  | 0.0714286 |                 0.748984 |                  0.798291 |                 0.877701 |            5 |

## 各模型最佳参数
### GLM
- best_cv_auc: 0.791818
- best_params: `{"classifier__C": 0.01, "classifier__penalty": "l2"}`

### RF
- best_cv_auc: 0.764722
- best_params: `{"class_weight": "balanced", "max_depth": 3, "max_features": "log2", "min_samples_leaf": 5, "min_samples_split": 2, "n_estimators": 800}`

### SVM
- best_cv_auc: 0.799299
- best_params: `{"classifier__C": 0.1, "classifier__gamma": 0.001, "classifier__kernel": "rbf"}`

### NNET
- best_cv_auc: 0.787766
- best_params: `{"classifier__activation": "tanh", "classifier__alpha": 0.01, "classifier__batch_size": 16, "classifier__hidden_layer_sizes": [32], "classifier__learning_rate_init": 0.01, "classifier__solver": "adam"}`

### GBM
- best_cv_auc: 0.741981
- best_params: `{"learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 200, "subsample": 1.0}`

### XGBoost
- best_cv_auc: 0.774361
- best_params: `{"colsample_bytree": 1.0, "gamma": 0.3, "learning_rate": 0.01, "max_depth": 2, "min_child_weight": 1, "n_estimators": 200, "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 1.0}`
