本项目用于建立头颈癌手术后肺部感染预测模型

1. 原始数据.xlsx 为医院给的数据，清洗后得到 data_deepclean.csv，因变量为肺部感染(PulmonaryInfection)
2. ML.py 构建机器学习模型用于预测分析，"OperationDurationMin""PreopConcurrentCRT""NeckDissection""IntraopTransfusion""Tracheostomy" 为筛选出的危险独立因素，作为因变量进行模型构建
3. 你需要完成的工作是：此数据阴阳样本比例极度不平衡，需要尝试权重调整/阈值调整/过采样等方法，使其保持特异性和敏感度的平衡。
4. 对于使用的方法，无论效果如何，都要分析原因
5. 另附参考文献一篇，以便你熟悉流程
6. 4.3（周五）晚10点前向姚老师提交汇报PPT，提交前可以给师姐看一眼。
