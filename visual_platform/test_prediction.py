#!/usr/bin/env python3
"""
测试预测功能的简单脚本
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from prediction_service import DeepSurvPredictor, PatientDataValidator
    print("✅ 预测服务模块导入成功")
    
    # 创建预测器实例
    predictor = DeepSurvPredictor()
    print("✅ 预测器创建成功")
    
    # 测试患者数据
    test_patient = {
        'age': 55,
        'tumor_size': 3.2,
        'stage': 2,
        'grade': 2,
        'lymph_nodes': 1,
        'er_positive': 1,
        'pr_positive': 1,
        'her2_positive': 0,
        'surgery_type': 2,
        'chemotherapy': 1,
        'radiotherapy': 1,
        'menopause_status': 1,
        'histology_type': 0
    }
    
    # 验证数据
    is_valid, errors = PatientDataValidator.validate_patient_data(test_patient)
    if is_valid:
        print("✅ 患者数据验证通过")
    else:
        print("❌ 患者数据验证失败:", errors)
        exit(1)
    
    # 测试风险预测
    risk_score = predictor.predict_risk_score(test_patient)
    print(f"✅ 风险评分预测成功: {risk_score:.3f}")
    
    # 测试风险解释
    interpretation = predictor.get_risk_interpretation(risk_score)
    print(f"✅ 风险解释生成成功: {interpretation['risk_level']}")
    
    # 测试生存概率预测
    time_points = [12, 24, 36, 60]
    survival_probs = predictor.predict_survival_probability(test_patient, time_points)
    print(f"✅ 生存概率预测成功:")
    for t, prob in zip(time_points, survival_probs):
        print(f"   {t}个月: {prob:.3f}")
    
    # 测试特征重要性
    feature_importance = predictor.get_feature_importance(test_patient)
    print(f"✅ 特征重要性计算成功: {len(feature_importance)} 个特征")
    
    print("\n🎉 所有测试通过！预测功能工作正常。")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()