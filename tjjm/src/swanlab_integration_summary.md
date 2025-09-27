# SwanLab 集成总结

## 概述
已为 tjjm/src 目录中的主要脚本添加了 SwanLab 支持，项目名称为 "sci"，实验名称为脚本名（去掉.py扩展名）。

## 已集成的脚本

### 1. 基础模型脚本
- ✅ `linear.py` - 线性回归模型
- ✅ `SVR.py` - 支持向量回归模型
- ✅ `xgb_model.py` - XGBoost模型
- ✅ `rf_model.py` - Random Forest模型
- ✅ `lgb_model.py` - LightGBM模型
- ✅ `neural_network.py` - 神经网络模型

### 2. 高级模型脚本
- ✅ `transformer.py` - Transformer模型
- ✅ `nn_based_autoencoder.py` - 基于神经网络的自动编码器
- ✅ `ngboost_model.py` - NGBoost模型
- ✅ `catboost_model.py` - CatBoost模型
- ✅ `cox_model.py` - Cox比例风险模型

### 3. 集成模型脚本
- ✅ `stacking_model_rf_xgb_lgb.py` - Stacking集成模型
- ✅ `Voting_model_rf_xgb_lgb.py` - Voting集成模型
- ✅ `bootstrap_敏感性分析.py` - Bootstrap敏感性分析

### 4. 其他模型脚本
- ✅ `KNN.py` - K近邻模型
- ✅ `bayes.py` - 贝叶斯回归模型
- ✅ `ridge.py` - 岭回归模型
- ✅ `lasso.py` - Lasso回归模型
- ✅ `ElasticNet.py` - 弹性网络模型
- ✅ `GPR.py` - 高斯过程回归模型
- ✅ `locally_weighted_regression.py` - 局部加权回归模型

## SwanLab 配置

### 配置文件
- `swanlab_config.py` - 统一的SwanLab配置模块

### 主要功能
1. **初始化**: `init_swanlab(script_name)` - 创建SwanLab运行实例
2. **指标记录**: `log_metrics(run, metrics_dict)` - 记录评估指标
3. **参数记录**: `log_model_params(run, model_params)` - 记录模型参数
4. **完成运行**: `finish_run(run)` - 完成SwanLab运行

### 记录的信息
- **项目名称**: "sci"
- **实验名称**: 脚本文件名（去掉.py）
- **时间戳**: 运行开始时间
- **Python版本**: 运行环境信息
- **模型参数**: 模型类型、超参数、数据维度等
- **评估指标**: MAE、RMSE、R²等性能指标

## 使用方法

### 1. 安装SwanLab
```bash
pip install swanlab
```

### 2. 运行脚本
```bash
cd /data1/chzhang/tjjm/src
python script_name.py
```

### 3. 查看结果
SwanLab会自动启动Web界面，显示实验记录和可视化结果。

## 注意事项

1. **数据预处理**: 所有脚本都使用统一的 `data_preprocessing_0()` 函数
2. **错误处理**: 添加了异常处理机制，确保SwanLab记录不会影响脚本运行
3. **兼容性**: 保持原有脚本功能不变，只是添加了SwanLab支持
4. **性能**: SwanLab记录是异步的，不会显著影响脚本运行速度

## 待处理脚本

以下脚本可能需要手动添加SwanLab支持（如果它们有特殊的结构或依赖）：
- `best_base_bagging_*.py` - 基础模型Bagging优化脚本
- `best_base_model_choose_*.py` - 基础模型选择脚本
- `boosting_stacking_model_voting_model_*.py` - 复杂的集成模型脚本
- `NODE.py` - 神经决策树模型
- `RuleFit.py` - 规则拟合模型

## 建议

1. 在运行新脚本前，建议先检查是否已添加SwanLab支持
2. 如果脚本运行出错，可以暂时注释掉SwanLab相关代码
3. 定期备份SwanLab的实验记录
4. 可以根据需要自定义记录的指标和参数
