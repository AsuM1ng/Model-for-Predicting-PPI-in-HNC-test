# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260327_171352_BJT`
- 训练集路径: `analysis_outputs\train_set.csv`
- 测试集路径: `analysis_outputs\test_set.csv`
- 特征数量: 5
- 特征列表: PreopConcurrentCRT, OperationDurationMin, Tracheostomy, NeckDissection, AlcoholHistory

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
| RF      |            0.377428  |                      0.7      |                      0.613405 |     0.713859 |      0.713805 |     0.144583 |   0.871992 |  0.621399  |      0.923077 |      0.615922 | 0.08      |                 0.768489 |                  0.843871 |                 0.878039 |            5 |
| GLM     |            0.464542  |                      0.666667 |                      0.635548 |     0.746781 |      0.743189 |     0.134904 |   0.871723 |  0.654321  |      0.846154 |      0.650838 | 0.080292  |                 0.779574 |                  0.860657 |                 0.883466 |            5 |
| SVM     |            0.0172947 |                      0.366667 |                      0.624177 |     0.75916  |      0.757954 |     0.125084 |   0.8685   |  0.0178326 |      1        |      0        | 0.0350404 |                 0.121793 |                  0.845617 |                 0.873617 |            5 |
| NNET    |            0.0165664 |                      0.733333 |                      0.636146 |     0.753313 |      0.737253 |     0.138604 |   0.862376 |  0.648834  |      0.846154 |      0.645251 | 0.0791367 |                 0.741494 |                  0.850774 |                 0.884465 |            5 |
| GBM     |            0.0136776 |                      0.7      |                      0.614004 |     0.712317 |      0.719196 |     0.156128 |   0.856575 |  0.58299   |      0.923077 |      0.576816 | 0.0731707 |                 0.73595  |                  0.843468 |                 0.887961 |            5 |
| XGBoost |            0.0176484 |                      0.766667 |                      0.602633 |     0.728565 |      0.720877 |     0.138863 |   0.844596 |  0.62963   |      0.923077 |      0.624302 | 0.0816327 |                 0.74747  |                  0.82652  |                 0.86167  |            5 |

## 各模型最佳参数
### GLM
- best_cv_auc: 0.746781
- best_params: `{"classifier__C": 0.01, "classifier__penalty": "l2"}`

### RF
- best_cv_auc: 0.713859
- best_params: `{"class_weight": "balanced", "max_depth": 3, "max_features": "log2", "min_samples_leaf": 5, "min_samples_split": 2, "n_estimators": 800}`

### SVM
- best_cv_auc: 0.759160
- best_params: `{"classifier__C": 0.1, "classifier__gamma": 0.001, "classifier__kernel": "rbf"}`

### NNET
- best_cv_auc: 0.753313
- best_params: `{"classifier__activation": "tanh", "classifier__alpha": 0.01, "classifier__batch_size": 16, "classifier__hidden_layer_sizes": [32], "classifier__learning_rate_init": 0.01, "classifier__solver": "adam"}`

### GBM
- best_cv_auc: 0.712317
- best_params: `{"learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 5, "n_estimators": 200, "subsample": 1.0}`

### XGBoost
- best_cv_auc: 0.728565
- best_params: `{"colsample_bytree": 1.0, "gamma": 0.3, "learning_rate": 0.01, "max_depth": 2, "min_child_weight": 3, "n_estimators": 200, "reg_alpha": 0.1, "reg_lambda": 5.0, "subsample": 1.0}`
