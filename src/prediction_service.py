"""
预测服务模块
为可视化平台提供DeepSurv模型预测功能
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSurvPredictor:
    """DeepSurv模型预测器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 preprocessor_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            preprocessor_path: 预处理器文件路径
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.feature_ranges = None
        self.is_loaded = False
        
        # 尝试加载模型和预处理器
        if model_path and preprocessor_path:
            self.load_model(model_path, preprocessor_path)
    
    def load_model(self, model_path: str, preprocessor_path: str) -> bool:
        """
        加载模型和预处理器
        
        Args:
            model_path: 模型文件路径
            preprocessor_path: 预处理器文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 加载预处理器
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # 检查是否有DeepSurv模型文件
            model_file = Path(model_path)
            if model_file.exists():
                # 如果有实际的模型文件，加载它
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # 如果没有实际模型，创建一个模拟模型
                logger.warning("DeepSurv模型文件不存在，使用模拟预测器")
                self.model = None
            
            # 设置特征信息
            self._setup_feature_info()
            
            self.is_loaded = True
            logger.info("预测器加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            self.is_loaded = False
            return False
    
    def _setup_feature_info(self):
        """设置特征信息"""
        # 定义常见的癌症生存分析特征
        self.feature_names = [
            'age', 'tumor_size', 'lymph_nodes', 'grade', 'stage',
            'er_positive', 'pr_positive', 'her2_positive', 
            'chemotherapy', 'radiotherapy', 'surgery_type',
            'menopause_status', 'histology_type'
        ]
        
        # 特征范围（用于输入验证）
        self.feature_ranges = {
            'age': (18, 100),
            'tumor_size': (0.1, 20.0),
            'lymph_nodes': (0, 50),
            'grade': (1, 3),
            'stage': (1, 4),
            'er_positive': (0, 1),
            'pr_positive': (0, 1),
            'her2_positive': (0, 1),
            'chemotherapy': (0, 1),
            'radiotherapy': (0, 1),
            'surgery_type': (0, 3),
            'menopause_status': (0, 2),
            'histology_type': (0, 5)
        }
    
    def preprocess_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        预处理输入数据
        
        Args:
            patient_data: 患者数据字典
            
        Returns:
            np.ndarray: 预处理后的特征向量
        """
        try:
            # 创建特征向量
            features = []
            
            # 按照预定义的特征顺序提取数据
            for feature_name in self.feature_names:
                if feature_name in patient_data:
                    value = patient_data[feature_name]
                    
                    # 验证特征值范围
                    if feature_name in self.feature_ranges:
                        min_val, max_val = self.feature_ranges[feature_name]
                        value = max(min_val, min(max_val, value))
                    
                    features.append(value)
                else:
                    # 使用默认值
                    features.append(0.0)
            
            # 转换为numpy数组
            feature_array = np.array(features).reshape(1, -1)
            
            # 如果有预处理器，使用它进行标准化
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                feature_array = self.preprocessor.transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"预处理输入数据失败: {str(e)}")
            raise
    
    def predict_risk_score(self, patient_data: Dict[str, Any]) -> float:
        """
        预测风险评分
        
        Args:
            patient_data: 患者数据字典
            
        Returns:
            float: 风险评分
        """
        try:
            # 预处理输入
            features = self.preprocess_input(patient_data)
            
            if self.model is not None:
                # 使用实际模型进行预测
                if hasattr(self.model, 'predict_risk'):
                    risk_score = self.model.predict_risk(features)[0]
                else:
                    # 其他模型接口
                    risk_score = self.model.predict(features)[0]
            else:
                # 使用基于规则的模拟预测
                risk_score = self._simulate_risk_prediction(patient_data)
            
            # 确保风险评分在合理范围内
            risk_score = max(0.0, min(1.0, float(risk_score)))
            
            return risk_score
            
        except Exception as e:
            logger.error(f"预测风险评分失败: {str(e)}")
            # 返回默认风险评分
            return 0.5
    
    def _simulate_risk_prediction(self, patient_data: Dict[str, Any]) -> float:
        """
        模拟风险预测（当没有实际模型时使用）
        
        Args:
            patient_data: 患者数据字典
            
        Returns:
            float: 模拟的风险评分
        """
        base_risk = 0.3
        
        # 年龄因子
        age = patient_data.get('age', 50)
        age_factor = (age - 50) * 0.008
        
        # 肿瘤大小因子
        tumor_size = patient_data.get('tumor_size', 2.0)
        tumor_factor = tumor_size * 0.015
        
        # 分期因子
        stage = patient_data.get('stage', 2)
        stage_factor = (stage - 1) * 0.1
        
        # 淋巴结因子
        lymph_nodes = patient_data.get('lymph_nodes', 0)
        lymph_factor = lymph_nodes * 0.02
        
        # 分级因子
        grade = patient_data.get('grade', 2)
        grade_factor = (grade - 1) * 0.05
        
        # 受体状态因子
        er_positive = patient_data.get('er_positive', 1)
        er_factor = -0.1 if er_positive else 0.05
        
        # 治疗因子
        chemotherapy = patient_data.get('chemotherapy', 1)
        chemo_factor = -0.08 if chemotherapy else 0.0
        
        radiotherapy = patient_data.get('radiotherapy', 1)
        radio_factor = -0.05 if radiotherapy else 0.0
        
        # 计算总风险
        total_risk = (base_risk + age_factor + tumor_factor + 
                     stage_factor + lymph_factor + grade_factor + 
                     er_factor + chemo_factor + radio_factor)
        
        # 添加一些随机性
        noise = np.random.normal(0, 0.02)
        total_risk += noise
        
        return max(0.0, min(1.0, total_risk))
    
    def predict_survival_probability(self, patient_data: Dict[str, Any], 
                                   time_points: List[float]) -> List[float]:
        """
        预测生存概率
        
        Args:
            patient_data: 患者数据字典
            time_points: 时间点列表
            
        Returns:
            List[float]: 各时间点的生存概率
        """
        try:
            # 获取风险评分
            risk_score = self.predict_risk_score(patient_data)
            
            # 基于风险评分计算生存概率
            survival_probs = []
            for t in time_points:
                # 使用指数衰减模型
                # 风险越高，生存概率衰减越快
                decay_rate = risk_score * 0.1  # 调整衰减速率
                survival_prob = np.exp(-decay_rate * t)
                survival_probs.append(max(0.0, min(1.0, survival_prob)))
            
            return survival_probs
            
        except Exception as e:
            logger.error(f"预测生存概率失败: {str(e)}")
            # 返回默认生存概率
            return [0.5] * len(time_points)
    
    def get_feature_importance(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """
        获取特征重要性（简化版）
        
        Args:
            patient_data: 患者数据字典
            
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        try:
            # 模拟特征重要性计算
            feature_importance = {}
            
            # 基于患者数据和预定义权重计算重要性
            weights = {
                'age': 0.15,
                'tumor_size': 0.20,
                'stage': 0.25,
                'lymph_nodes': 0.18,
                'grade': 0.12,
                'er_positive': 0.10
            }
            
            for feature, weight in weights.items():
                if feature in patient_data:
                    # 特征值的标准化影响
                    value = patient_data[feature]
                    if feature == 'age':
                        normalized_impact = (value - 50) / 50
                    elif feature == 'tumor_size':
                        normalized_impact = value / 10
                    elif feature == 'stage':
                        normalized_impact = (value - 1) / 3
                    elif feature == 'lymph_nodes':
                        normalized_impact = value / 20
                    elif feature == 'grade':
                        normalized_impact = (value - 1) / 2
                    elif feature == 'er_positive':
                        normalized_impact = -0.5 if value else 0.5
                    else:
                        normalized_impact = 0
                    
                    importance = weight * abs(normalized_impact)
                    feature_importance[feature] = importance
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"计算特征重要性失败: {str(e)}")
            return {}
    
    def get_risk_interpretation(self, risk_score: float) -> Dict[str, Any]:
        """
        解释风险评分
        
        Args:
            risk_score: 风险评分
            
        Returns:
            Dict[str, Any]: 风险解释
        """
        interpretation = {
            'risk_score': risk_score,
            'risk_level': '',
            'risk_category': '',
            'color': '',
            'description': '',
            'recommendations': []
        }
        
        if risk_score < 0.3:
            interpretation.update({
                'risk_level': '低风险',
                'risk_category': 'low',
                'color': 'green',
                'description': '患者的预测风险较低，预后相对良好。',
                'recommendations': [
                    '定期随访观察',
                    '保持健康生活方式',
                    '按医嘱进行常规检查'
                ]
            })
        elif risk_score < 0.7:
            interpretation.update({
                'risk_level': '中等风险',
                'risk_category': 'medium',
                'color': 'orange',
                'description': '患者的预测风险处于中等水平，需要密切关注。',
                'recommendations': [
                    '加强定期监测',
                    '考虑辅助治疗',
                    '保持良好的生活习惯',
                    '心理支持和指导'
                ]
            })
        else:
            interpretation.update({
                'risk_level': '高风险',
                'risk_category': 'high',
                'color': 'red',
                'description': '患者的预测风险较高，需要积极的治疗和监护。',
                'recommendations': [
                    '制定积极的治疗方案',
                    '频繁的医学监测',
                    '考虑多学科会诊',
                    '提供心理支持',
                    '家属参与护理决策'
                ]
            })
        
        return interpretation


class PatientDataValidator:
    """患者数据验证器"""
    
    @staticmethod
    def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证患者数据
        
        Args:
            data: 患者数据字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误消息列表)
        """
        errors = []
        
        # 必需字段
        required_fields = ['age', 'tumor_size', 'stage']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"缺少必需字段: {field}")
        
        # 数值范围验证
        if 'age' in data:
            age = data['age']
            if not (18 <= age <= 100):
                errors.append("年龄必须在18-100岁之间")
        
        if 'tumor_size' in data:
            size = data['tumor_size']
            if not (0.1 <= size <= 20.0):
                errors.append("肿瘤大小必须在0.1-20.0cm之间")
        
        if 'stage' in data:
            stage = data['stage']
            if stage not in [1, 2, 3, 4]:
                errors.append("癌症分期必须是1-4之间的整数")
        
        return len(errors) == 0, errors