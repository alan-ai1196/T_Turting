# 癌症生存分析模型评估增强版

## 🆕 新增功能

本次更新为癌症生存分析项目增加了更全面的模型评估指标和可视化功能：

### 1. 🎯 Brier Score 时间依赖评估
- **功能**: 评估模型在不同时间点的预测准确性
- **优势**: 提供比C-index更细粒度的性能分析
- **解释**: 值越低表示预测越准确（范围0-1）

### 2. 📊 集成Brier Score (IBS) 
- **功能**: 整个时间范围内的综合预测性能评估
- **优势**: 单一指标综合评估模型整体表现
- **计算**: Brier Score在整个随访期间的积分

### 3. 📈 增强Kaplan-Meier生存曲线
- **新功能**:
  - 多模型风险分层对比
  - Log-rank统计检验p值显示
  - 中位生存时间标注
  - 置信区间展示
  - 综合对比视图

### 4. 🎨 Streamlit可视化平台增强
- **多标签页设计**: C-index对比、Brier Score分析、综合评估、性能排名
- **雷达图**: 多维度性能可视化
- **交互式图表**: Plotly图表支持缩放和悬停
- **详细指标说明**: 帮助用户理解各评估指标

## 📁 新增文件

### 代码文件
- `src/model_evaluation.py` - 增强评估功能
- `notebooks/model_evaluation.ipynb` - 完整评估流程
- `visual_platform/streamlit_app.py` - 更新可视化界面

### 输出文件
- `data/processed/brier_scores_results.csv` - Brier Score详细结果
- `data/processed/integrated_brier_scores_results.csv` - IBS结果
- `reports/comprehensive_survival_analysis.png` - 综合生存分析图
- `reports/comprehensive_evaluation_summary.png` - 评估总结图
- `reports/evaluation_summary_report.md` - 详细评估报告

## 🔧 使用方法

### 1. 运行增强评估
```bash
# 在notebooks目录下运行
jupyter notebook model_evaluation.ipynb
```

### 2. 启动可视化平台
```bash
cd visual_platform
./run.sh
```

### 3. 查看评估结果
- 浏览器访问: http://localhost:8501
- 选择"模型对比"页面查看新增的评估指标

## 📊 评估指标对比

| 指标 | 范围 | 最优值 | 特点 |
|------|------|--------|------|
| C-index | 0.5-1.0 | 越高越好 | 排序一致性，经典指标 |
| Brier Score | 0-1 | 越低越好 | 时间依赖，预测准确性 |
| IBS | 0-1 | 越低越好 | 综合性能，整体评估 |
| Log-rank p值 | 0-1 | <0.05显著 | 风险分层统计检验 |

## 🎯 评估流程

1. **数据加载**: 加载模型预测结果
2. **C-index计算**: 传统一致性指数评估
3. **Brier Score分析**: 多时间点预测准确性
4. **IBS计算**: 综合预测性能评估
5. **生存曲线绘制**: K-M曲线风险分层分析
6. **综合评估**: 多指标综合排名和性能等级
7. **结果保存**: 所有结果保存为CSV和图片

## 🔬 技术实现

### Brier Score计算
```python
# 时间依赖的Brier Score
for t in time_points:
    survival_probs = model.predict_survival_probability(X, t)
    bs = brier_score(y_true, survival_probs, t)
```

### IBS计算
```python
# 集成Brier Score
time_points = np.linspace(0, max_time, 50)
survival_functions = model.predict_survival_function(X, time_points)
ibs = integrated_brier_score(y_true, survival_functions, time_points)
```

### K-M曲线增强
```python
# 风险分层生存曲线
risk_groups = create_risk_groups(risk_scores)
for group in [low, medium, high]:
    kmf.fit(durations[group], events[group])
    kmf.plot(label=f'Risk Group {group}')
```

## 📈 性能解释

### C-index性能等级
- **优秀** (>0.7): 模型具有很好的区分能力
- **良好** (0.6-0.7): 模型有一定的预测价值
- **一般** (<0.6): 模型性能接近随机预测

### Brier Score解释
- **<0.15**: 优秀的预测准确性
- **0.15-0.25**: 良好的预测准确性
- **>0.25**: 预测准确性有待提升

### 风险分层评估
- **Log-rank p<0.05**: 风险分层具有统计学意义
- **生存曲线分离度**: 反映模型的临床实用性

## 🔍 结果解读指南

### 1. 查看C-index对比
- 关注最高的C-index值
- 比较模型间的差异程度
- 参考性能等级标准

### 2. 分析Brier Score趋势
- 观察不同时间点的变化
- 识别模型在哪个时间段预测更准确
- 比较平均Brier Score

### 3. 评估IBS综合性能
- IBS提供整体性能排名
- 结合C-index和Brier Score综合判断
- 考虑临床应用场景

### 4. 解读生存曲线
- 检查风险分层的分离程度
- 关注Log-rank检验的p值
- 比较不同模型的风险分层能力

## 🚀 未来改进方向

1. **可解释性分析**: 添加SHAP值和特征重要性分析
2. **时间依赖ROC**: 添加时间依赖的ROC分析
3. **校准曲线**: 评估预测概率的校准性
4. **临床净获益**: 添加决策曲线分析
5. **交叉验证**: 增加更稳健的模型验证方法

## 📞 技术支持

如有问题或建议，请：
1. 检查数据文件是否完整
2. 确认Python环境和依赖包
3. 查看错误日志和提示信息
4. 参考评估流程步骤

---

*增强版评估功能为癌症生存分析提供了更全面、更准确的模型评估体系，助力临床决策和科研工作。*