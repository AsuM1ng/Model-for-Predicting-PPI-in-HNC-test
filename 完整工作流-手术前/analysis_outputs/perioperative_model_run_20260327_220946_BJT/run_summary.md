# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260327_220946_BJT`
- 训练集路径: `analysis_outputs\train_set.csv`
- 测试集路径: `analysis_outputs\test_set.csv`
- 特征数量: 3
- 特征列表: PreopConcurrentCRT, PreopHGB, AlcoholHistory

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
| GLM     |            0.474489  |                      0.733333 |                      0.62451  |     0.741903 |      0.745073 |     0.1308   |   0.667335 |   0.607692 |      0.692308 |      0.606258 | 0.0555556 |                 0.622972 |                  0.662822 |                 0.674438 |            3 |
| SVM     |            0.0184513 |                      0.2      |                      0.640739 |     0.74287  |      0.745983 |     0.130758 |   0.666734 |   0.983333 |      0        |      1        | 0         |                 0.328427 |                  0.341089 |                 0.66823  |            3 |
| GBM     |            0.0126029 |                      0.7      |                      0.633464 |     0.679215 |      0.686964 |     0.147446 |   0.65866  |   0.652564 |      0.692308 |      0.65189  | 0.0622837 |                 0.573196 |                  0.636496 |                 0.699064 |            3 |
| NNET    |            0.0193814 |                      0.666667 |                      0.628428 |     0.718819 |      0.711649 |     0.162696 |   0.65159  |   0.866667 |      0.384615 |      0.874837 | 0.0877193 |                 0.633332 |                  0.711664 |                 0.741766 |            3 |
| RF      |            0.387318  |                      0.766667 |                      0.669838 |     0.66903  |      0.657521 |     0.160968 |   0.642914 |   0.684615 |      0.692308 |      0.684485 | 0.0681818 |                 0.548644 |                  0.625138 |                 0.715405 |            3 |
| XGBoost |            0.0141627 |                      0.733333 |                      0.601567 |     0.702058 |      0.701394 |     0.150238 |   0.628422 |   0.666667 |      0.692308 |      0.666232 | 0.0647482 |                 0.587123 |                  0.617165 |                 0.641821 |            3 |

## 各模型最佳参数
### GLM
- best_cv_auc: 0.741903
- best_params: `{"classifier__C": 0.5, "classifier__penalty": "l2"}`

### RF
- best_cv_auc: 0.669030
- best_params: `{"class_weight": "balanced_subsample", "max_depth": 3, "max_features": "sqrt", "min_samples_leaf": 3, "min_samples_split": 5, "n_estimators": 800}`

### SVM
- best_cv_auc: 0.742870
- best_params: `{"classifier__C": 0.1, "classifier__gamma": 0.001, "classifier__kernel": "rbf"}`

### NNET
- best_cv_auc: 0.718819
- best_params: `{"classifier__activation": "tanh", "classifier__alpha": 0.0001, "classifier__batch_size": 64, "classifier__hidden_layer_sizes": [32, 16], "classifier__learning_rate_init": 0.005, "classifier__solver": "adam"}`

### GBM
- best_cv_auc: 0.679215
- best_params: `{"learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 200, "subsample": 1.0}`

### XGBoost
- best_cv_auc: 0.702058
- best_params: `{"colsample_bytree": 0.6, "gamma": 0.3, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5, "n_estimators": 300, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.6}`
