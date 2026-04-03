# 建模结果信息文档

- 运行目录: `analysis_outputs\perioperative_model_run_20260324_234415_BJT`
- 训练集路径: `analysis_outputs\train_set.csv`
- 测试集路径: `analysis_outputs\test_set.csv`
- 特征数量: 5
- 特征列表: OperationDurationMin, PreopConcurrentCRT, NeckDissection, IntraopTransfusion, Tracheostomy

## 关键参数
- RANDOM_STATE: 15
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
| GBM     |            0.0111279 |                      0.827586 |                      0.620215 |     0.750783 |      0.749994 |     0.162272 |   0.889644 |   0.652949 |      0.916667 |      0.648536 | 0.08      |                 0.713629 |                  0.823774 |                 0.882088 |            5 |
| XGBoost |            0.012769  |                      0.827586 |                      0.619617 |     0.764673 |      0.761779 |     0.153998 |   0.873722 |   0.621399 |      0.916667 |      0.616457 | 0.0738255 |                 0.775742 |                  0.842486 |                 0.894843 |            5 |
| GLM     |            0.42246   |                      0.827586 |                      0.613636 |     0.788563 |      0.795293 |     0.139395 |   0.83897  |   0.577503 |      0.916667 |      0.571827 | 0.0666667 |                 0.791306 |                  0.832636 |                 0.87588  |            5 |
| NNET    |            0.0839613 |                      0.689655 |                      0.748206 |     0.800458 |      0.802035 |     0.131086 |   0.837576 |   0.824417 |      0.666667 |      0.827057 | 0.111111  |                 0.825692 |                  0.852685 |                 0.870197 |            5 |
| RF      |            0.363607  |                      0.827586 |                      0.620215 |     0.774072 |      0.771418 |     0.15061  |   0.826127 |   0.607682 |      0.916667 |      0.60251  | 0.0714286 |                 0.757611 |                  0.810611 |                 0.878946 |            5 |
| SVM     |            0.0153384 |                      0.793103 |                      0.653708 |     0.766524 |      0.76207  |     0.149191 |   0.73367  |   0.679012 |      0.666667 |      0.679219 | 0.064     |                 0.555137 |                  0.78417  |                 0.883487 |            5 |

## 各模型最佳参数
### GLM
- best_cv_auc: 0.788563
- best_params: `{"classifier__C": 0.01, "classifier__penalty": "l2"}`

### RF
- best_cv_auc: 0.774072
- best_params: `{"class_weight": "balanced", "max_depth": 3, "max_features": "log2", "min_samples_leaf": 5, "min_samples_split": 20, "n_estimators": 1000}`

### SVM
- best_cv_auc: 0.766524
- best_params: `{"classifier__C": 10.0, "classifier__gamma": 0.1, "classifier__kernel": "linear"}`

### NNET
- best_cv_auc: 0.800458
- best_params: `{"classifier__activation": "tanh", "classifier__alpha": 0.1, "classifier__batch_size": 16, "classifier__hidden_layer_sizes": [64, 32], "classifier__learning_rate_init": 0.0005, "classifier__solver": "adam"}`

### GBM
- best_cv_auc: 0.750783
- best_params: `{"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 3, "n_estimators": 100, "subsample": 1.0}`

### XGBoost
- best_cv_auc: 0.764673
- best_params: `{"colsample_bytree": 1.0, "gamma": 1.0, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 1, "n_estimators": 300, "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.8}`
