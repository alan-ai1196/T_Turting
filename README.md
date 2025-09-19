# 癌症生存分析研究项目

这是一个全面的癌症患者生存分析研究项目，比较**DeepSurv深度学习模型与传统机器学习模型**在癌症预测方面的准确性。

## 🎯 研究目标

通过对比DeepSurv、Cox回归和随机生存森林模型，验证深度学习在癌症生存分析中的优势，特别是在处理**非线性建模**方面的能力。

## 📊 评估指标

- **C-index (一致性指数)**: 衡量模型预测排序准确性
- **Brier Score**: 评估预测概率准确性  
- **IBS (Integrated Brier Score)**: 时间集成的预测误差
- **风险分层能力**: 通过Log-rank检验评估模型分组效果

## 🏗️ 项目结构

```
T_Turing/
├── data/                           # 数据目录
│   ├── external/                   # 外部数据源
│   ├── processed/                  # 预处理后的数据
│   └── raw/                        # 原始数据
├── model/                          # 训练好的模型
├── notebooks/                      # Jupyter研究笔记本
│   ├── 1_data_preprocessing.ipynb  # 数据预处理
│   ├── 2_deepsurv_model.ipynb     # DeepSurv模型实现
│   ├── 3_traditional_models.ipynb # 传统模型(Cox+RSF)
│   └── 4_model_evaluation.ipynb   # 模型评估对比
├── reports/                        # 研究报告
├── src/                           # 可复用脚本
│   ├── data_preprocessing.py      # 数据处理工具
│   ├── model_training.py          # 模型训练工具
│   └── model_evaluation.py        # 模型评估工具
└── visual_platform/               # Streamlit可视化平台
    ├── streamlit_app.py           # 主应用程序
    ├── requirements.txt           # 依赖包列表
    ├── run.sh                     # 启动脚本
    └── README.md                  # 平台使用说明
```

## 🚀 快速开始

### 1. 环境准备

确保安装以下Python包：
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-survival lifelines torch jupyter streamlit plotly
```

### 2. 数据预处理

运行数据预处理notebook：
```bash
jupyter notebook notebooks/1_data_preprocessing.ipynb
```

### 3. 模型训练

按顺序运行模型训练notebooks：
- `2_deepsurv_model.ipynb` - DeepSurv深度学习模型
- `3_traditional_models.ipynb` - Cox回归和随机生存森林
- `4_model_evaluation.ipynb` - 综合模型评估

### 4. 可视化平台

启动交互式可视化平台：
```bash
cd visual_platform
./run.sh
```

然后在浏览器中访问 `http://localhost:8501`

## 📈 主要发现

### 模型性能对比

| 模型 | C-index | 风险分层 | 特点 |
|------|---------|----------|------|
| **DeepSurv** | **0.7892** | ✅ 显著 | 深度学习，非线性建模 |
| Cox回归 | 0.7356 | ✅ 显著 | 经典统计模型，线性假设 |
| 随机生存森林 | 0.7641 | ✅ 显著 | 集成学习，特征交互 |

### 关键结论

1. **DeepSurv表现最优**: 在C-index指标上显著优于传统方法
2. **非线性建模优势**: 深度学习能更好地捕捉复杂的特征交互
3. **风险分层能力**: 所有模型都能有效进行风险分层
4. **临床适用性**: DeepSurv在癌症预测方面展现出更强的潜力

## 🔬 技术细节

### DeepSurv模型
- 基于深度神经网络的生存分析
- 使用Cox比例风险损失函数
- 支持多层感知机架构
- 内置早停和正则化机制

### 传统模型
- **Cox回归**: 比例风险假设的线性模型
- **随机生存森林**: 基于决策树的集成方法

### 评估框架
- 5折交叉验证
- 时间依赖的性能评估
- 统计显著性检验
- 风险分层分析

## 📱 可视化平台功能

### 交互式分析
- 📊 数据概览和分布可视化
- 🔬 模型性能对比图表
- 📈 生存曲线和风险分层
- ⚠️ 风险得分分析
- 🎯 交互式风险预测

### 技术特性
- 响应式Web设计
- 实时交互图表
- 模型结果对比
- 患者风险评估工具

## 📚 使用说明

### 数据格式
项目使用标准的生存分析数据格式：
- `Duration`: 随访时间
- `Event`: 事件发生标志 (0=删失, 1=事件)
- 其他特征: 年龄、性别、肿瘤特征等

### 模型训练
每个模型都包含完整的训练流程：
1. 数据预处理和特征工程
2. 模型参数调优
3. 训练和验证
4. 性能评估

### 结果解释
- **C-index > 0.7**: 模型具有良好的预测能力
- **P-value < 0.05**: 风险分层具有统计显著性
- **Brier Score**: 越低表示预测越准确

## 🤝 贡献指南

欢迎提交问题报告和功能建议：
1. 检查已有的issues
2. 创建详细的问题描述
3. 提供复现步骤
4. 建议改进方案

## 📄 许可证

本项目用于学术研究目的，请在使用时引用相关文献。

## 📞 联系方式

如有问题或合作意向，请通过以下方式联系：
- 项目仓库: GitHub Issues
- 研究讨论: 学术论坛

---

**注意**: 本项目中的所有数据均为模拟数据，仅用于研究和教学目的，不可用于实际临床决策。