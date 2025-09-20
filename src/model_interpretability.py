"""
模型可解释性分析模块
提供生存分析模型的可解释性功能，包括SHAP分析、特征重要性等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
import pickle

warnings.filterwarnings('ignore')

class SurvivalModelExplainer:
    """生存分析模型可解释性分析器"""
    
    def __init__(self):
        self.models = {}
        self.data = {}
        self.feature_names = []
        self.shap_values = {}
        self.feature_importance = {}
        
    def load_models_and_data(self, model_dir, data_dir):
        """加载模型和数据"""
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)
        
        # 加载数据
        try:
            self.data['train'] = pd.read_csv(data_dir / 'train_data.csv')
            self.data['test'] = pd.read_csv(data_dir / 'test_data.csv')
            
            # 加载预处理器
            with open(data_dir / 'preprocessors.pkl', 'rb') as f:
                preprocessors = pickle.load(f)
            self.feature_names = preprocessors['feature_columns']
            self.preprocessors = preprocessors
            
            print(f"✓ 数据加载成功，特征数量: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        
        # 加载模型（如果存在）
        try:
            # 尝试加载DeepSurv模型
            if (model_dir / 'deepsurv_model.pth').exists():
                self.models['deepsurv'] = torch.load(model_dir / 'deepsurv_model.pth')
                print("✓ DeepSurv模型加载成功")
            
            # 尝试加载Cox模型
            if (model_dir / 'cox_model.pkl').exists():
                with open(model_dir / 'cox_model.pkl', 'rb') as f:
                    self.models['cox'] = pickle.load(f)
                print("✓ Cox模型加载成功")
            
            # 尝试加载RSF模型
            if (model_dir / 'rsf_model.pkl').exists():
                with open(model_dir / 'rsf_model.pkl', 'rb') as f:
                    self.models['rsf'] = pickle.load(f)
                print("✓ RSF模型加载成功")
                
        except Exception as e:
            print(f"模型加载警告: {e}")
            print("将使用预测结果进行可解释性分析")
        
        return True
    
    def prepare_data_for_analysis(self, dataset='test', sample_size=1000):
        """准备用于分析的数据"""
        if dataset not in self.data:
            print(f"数据集 {dataset} 不存在")
            return None
        
        data = self.data[dataset].copy()
        
        # 如果数据太大，进行采样
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
            print(f"数据采样至 {sample_size} 条记录")
        
        # 准备特征矩阵
        feature_data = data[self.feature_names].copy()
        
        # 准备目标变量
        target_data = {
            'duration': data['Duration'].values,
            'event': data['Event'].values
        }
        
        return feature_data, target_data
    
    def analyze_cox_model_interpretability(self):
        """分析Cox模型的可解释性"""
        if 'cox' not in self.models:
            print("Cox模型未加载，尝试从预测结果分析")
            return self._analyze_from_predictions('cox')
        
        cox_model = self.models['cox']
        
        # 获取回归系数（特征重要性）
        if hasattr(cox_model, 'params_'):
            coefficients = cox_model.params_
            feature_importance = pd.DataFrame({
                'feature': coefficients.index,
                'coefficient': coefficients.values,
                'abs_coefficient': np.abs(coefficients.values),
                'hazard_ratio': np.exp(coefficients.values)
            })
            
            feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
            
            self.feature_importance['cox'] = feature_importance
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 8))
            
            # 前20个最重要的特征
            top_features = feature_importance.head(20)
            
            plt.subplot(2, 2, 1)
            plt.barh(range(len(top_features)), top_features['abs_coefficient'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('绝对系数值')
            plt.title('Cox模型特征重要性 (绝对值)')
            plt.gca().invert_yaxis()
            
            plt.subplot(2, 2, 2)
            plt.barh(range(len(top_features)), top_features['coefficient'], 
                    color=['red' if x < 0 else 'blue' for x in top_features['coefficient']])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('回归系数')
            plt.title('Cox模型回归系数 (正负影响)')
            plt.gca().invert_yaxis()
            
            plt.subplot(2, 2, 3)
            plt.barh(range(len(top_features)), top_features['hazard_ratio'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('风险比 (Hazard Ratio)')
            plt.title('Cox模型风险比')
            plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
            plt.gca().invert_yaxis()
            
            plt.subplot(2, 2, 4)
            # 系数分布
            plt.hist(feature_importance['coefficient'], bins=30, alpha=0.7)
            plt.xlabel('回归系数')
            plt.ylabel('频数')
            plt.title('Cox模型系数分布')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('../reports/cox_interpretability.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        
        return None
    
    def analyze_rsf_model_interpretability(self):
        """分析随机生存森林模型的可解释性"""
        if 'rsf' not in self.models:
            print("RSF模型未加载，尝试使用替代方法分析")
            return self._analyze_rsf_from_predictions()
        
        rsf_model = self.models['rsf']
        
        # 获取特征重要性
        if hasattr(rsf_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rsf_model.feature_importances_
            })
            
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            self.feature_importance['rsf'] = feature_importance
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 10))
            
            # 前20个最重要的特征
            top_features = feature_importance.head(20)
            
            plt.subplot(2, 2, 1)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('重要性得分')
            plt.title('RSF模型特征重要性')
            plt.gca().invert_yaxis()
            
            plt.subplot(2, 2, 2)
            # 重要性分布
            plt.hist(feature_importance['importance'], bins=30, alpha=0.7)
            plt.xlabel('重要性得分')
            plt.ylabel('频数')
            plt.title('RSF特征重要性分布')
            
            plt.subplot(2, 2, 3)
            # 累积重要性
            cumsum_importance = np.cumsum(feature_importance['importance'])
            plt.plot(range(len(cumsum_importance)), cumsum_importance)
            plt.xlabel('特征数量')
            plt.ylabel('累积重要性')
            plt.title('RSF特征重要性累积分布')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            # 前10个特征的饼图
            top_10 = feature_importance.head(10)
            other_importance = feature_importance.iloc[10:]['importance'].sum()
            
            pie_data = list(top_10['importance']) + [other_importance]
            pie_labels = list(top_10['feature']) + ['其他特征']
            
            plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
            plt.title('RSF前10重要特征占比')
            
            plt.tight_layout()
            plt.savefig('../reports/rsf_interpretability.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        
        return None
    
    def analyze_deepsurv_interpretability_with_shap(self):
        """使用SHAP分析DeepSurv模型的可解释性"""
        print("=== DeepSurv SHAP分析 ===")
        
        # 准备数据
        X_data, y_data = self.prepare_data_for_analysis(sample_size=500)  # SHAP计算较慢，使用较小样本
        
        if X_data is None:
            return None
        
        try:
            # 创建DeepSurv包装器用于SHAP分析
            deepsurv_wrapper = DeepSurvWrapper(self.models.get('deepsurv'), self.feature_names)
            
            # 如果没有模型，使用预测结果进行分析
            if 'deepsurv' not in self.models:
                print("使用预测结果进行SHAP分析...")
                return self._analyze_deepsurv_from_predictions()
            
            # 创建SHAP解释器
            background = X_data.sample(n=100, random_state=42)  # 背景数据集
            explainer = shap.KernelExplainer(deepsurv_wrapper.predict, background)
            
            # 计算SHAP值
            print("计算SHAP值中...这可能需要几分钟时间")
            sample_data = X_data.sample(n=50, random_state=42)  # 解释样本
            shap_values = explainer.shap_values(sample_data)
            
            self.shap_values['deepsurv'] = {
                'values': shap_values,
                'data': sample_data,
                'expected_value': explainer.expected_value
            }
            
            # 可视化SHAP结果
            self._plot_shap_analysis(shap_values, sample_data, 'DeepSurv')
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP分析出错: {e}")
            print("尝试使用简化的可解释性方法...")
            return self._analyze_deepsurv_simplified()
    
    def _analyze_deepsurv_simplified(self):
        """DeepSurv的简化可解释性分析"""
        # 使用排列重要性分析
        X_data, y_data = self.prepare_data_for_analysis()
        
        if X_data is None:
            return None
        
        try:
            # 创建一个简单的替代模型来估计特征重要性
            from sklearn.ensemble import RandomForestRegressor
            
            # 加载DeepSurv的预测结果作为目标
            pred_file = Path('../data/processed/deepsurv_predictions.csv')
            if pred_file.exists():
                pred_data = pd.read_csv(pred_file)
                risk_scores = pred_data['Risk_Score'].values
                
                # 使用随机森林拟合特征到风险得分的关系
                rf_surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_surrogate.fit(X_data, risk_scores)
                
                # 获取特征重要性
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': rf_surrogate.feature_importances_
                })
                
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                self.feature_importance['deepsurv'] = feature_importance
                
                # 可视化
                self._plot_deepsurv_feature_importance(feature_importance)
                
                return feature_importance
            
        except Exception as e:
            print(f"简化分析也出错: {e}")
            return None
    
    def _plot_deepsurv_feature_importance(self, feature_importance):
        """绘制DeepSurv特征重要性图"""
        plt.figure(figsize=(15, 10))
        
        # 前20个最重要的特征
        top_features = feature_importance.head(20)
        
        plt.subplot(2, 3, 1)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性得分')
        plt.title('DeepSurv特征重要性 (代理模型)')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 3, 2)
        # 重要性分布
        plt.hist(feature_importance['importance'], bins=30, alpha=0.7)
        plt.xlabel('重要性得分')
        plt.ylabel('频数')
        plt.title('特征重要性分布')
        
        plt.subplot(2, 3, 3)
        # 累积重要性
        cumsum_importance = np.cumsum(feature_importance['importance'])
        plt.plot(range(len(cumsum_importance)), cumsum_importance)
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征重要性累积分布')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        # 前15个特征的饼图
        top_15 = feature_importance.head(15)
        other_importance = feature_importance.iloc[15:]['importance'].sum()
        
        pie_data = list(top_15['importance']) + [other_importance]
        pie_labels = list(top_15['feature']) + ['其他特征']
        
        plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
        plt.title('DeepSurv前15重要特征占比')
        
        plt.subplot(2, 3, 5)
        # 特征重要性的箱线图分析
        importance_categories = pd.cut(feature_importance['importance'], 
                                     bins=5, labels=['很低', '低', '中', '高', '很高'])
        category_counts = importance_categories.value_counts()
        plt.bar(category_counts.index, category_counts.values)
        plt.xlabel('重要性类别')
        plt.ylabel('特征数量')
        plt.title('特征重要性类别分布')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 6)
        # 前10特征与后10特征的对比
        top_10_mean = feature_importance.head(10)['importance'].mean()
        bottom_10_mean = feature_importance.tail(10)['importance'].mean()
        
        plt.bar(['前10特征', '后10特征'], [top_10_mean, bottom_10_mean], 
               color=['red', 'blue'], alpha=0.7)
        plt.ylabel('平均重要性')
        plt.title('重要特征vs非重要特征对比')
        
        plt.tight_layout()
        plt.savefig('../reports/deepsurv_interpretability.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_feature_importance_across_models(self):
        """比较不同模型的特征重要性"""
        if not self.feature_importance:
            print("请先运行各模型的可解释性分析")
            return None
        
        # 合并所有模型的特征重要性
        comparison_data = []
        
        for model_name, importance_df in self.feature_importance.items():
            if model_name == 'cox':
                # Cox模型使用绝对系数值
                for _, row in importance_df.iterrows():
                    comparison_data.append({
                        'model': model_name,
                        'feature': row['feature'],
                        'importance': row['abs_coefficient'],
                        'rank': importance_df.index[importance_df['feature'] == row['feature']].tolist()[0] + 1
                    })
            else:
                # 其他模型使用重要性得分
                for _, row in importance_df.iterrows():
                    comparison_data.append({
                        'model': model_name,
                        'feature': row['feature'],
                        'importance': row['importance'],
                        'rank': importance_df.index[importance_df['feature'] == row['feature']].tolist()[0] + 1
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 找出所有模型都认为重要的特征
        feature_counts = comparison_df.groupby('feature')['model'].count()
        common_features = feature_counts[feature_counts == len(self.feature_importance)].index.tolist()
        
        if len(common_features) > 0:
            print(f"所有模型都包含的特征数量: {len(common_features)}")
            
            # 可视化特征重要性对比
            self._plot_feature_importance_comparison(comparison_df, common_features)
        
        return comparison_df
    
    def _plot_feature_importance_comparison(self, comparison_df, common_features):
        """绘制特征重要性对比图"""
        plt.figure(figsize=(16, 12))
        
        # 选择前15个共同特征进行对比
        top_common_features = common_features[:15] if len(common_features) >= 15 else common_features
        
        # 1. 热力图展示不同模型的特征重要性
        plt.subplot(2, 3, 1)
        heatmap_data = comparison_df[comparison_df['feature'].isin(top_common_features)]
        pivot_data = heatmap_data.pivot(index='feature', columns='model', values='importance')
        
        # 标准化重要性得分（0-1范围）
        pivot_data_normalized = pivot_data.div(pivot_data.max(axis=0), axis=1)
        
        sns.heatmap(pivot_data_normalized, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('特征重要性热力图 (标准化)')
        plt.xlabel('模型')
        plt.ylabel('特征')
        
        # 2. 特征重要性排名对比
        plt.subplot(2, 3, 2)
        rank_data = comparison_df[comparison_df['feature'].isin(top_common_features)]
        rank_pivot = rank_data.pivot(index='feature', columns='model', values='rank')
        
        sns.heatmap(rank_pivot, annot=True, cmap='RdYlBu_r', fmt='.0f')
        plt.title('特征重要性排名热力图')
        plt.xlabel('模型')
        plt.ylabel('特征')
        
        # 3. 各模型top10特征的重叠分析
        plt.subplot(2, 3, 3)
        top_features_by_model = {}
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]
            top_10 = model_data.nsmallest(10, 'rank')['feature'].tolist()
            top_features_by_model[model] = set(top_10)
        
        # 计算模型间的Jaccard相似度
        models = list(top_features_by_model.keys())
        similarity_matrix = np.zeros((len(models), len(models)))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = len(top_features_by_model[model1] & top_features_by_model[model2])
                    union = len(top_features_by_model[model1] | top_features_by_model[model2])
                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        sns.heatmap(similarity_matrix, annot=True, xticklabels=models, yticklabels=models,
                   cmap='Blues', fmt='.3f')
        plt.title('模型间特征选择相似度 (Top10)')
        
        # 4. 特征重要性分布对比
        plt.subplot(2, 3, 4)
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]
            plt.hist(model_data['importance'], alpha=0.5, label=model, bins=20)
        
        plt.xlabel('特征重要性')
        plt.ylabel('频数')
        plt.title('各模型特征重要性分布')
        plt.legend()
        
        # 5. 一致性最高的特征
        plt.subplot(2, 3, 5)
        # 计算每个特征在所有模型中的平均排名
        avg_ranks = comparison_df.groupby('feature')['rank'].mean().sort_values()
        top_consistent = avg_ranks.head(10)
        
        plt.barh(range(len(top_consistent)), top_consistent.values)
        plt.yticks(range(len(top_consistent)), top_consistent.index)
        plt.xlabel('平均排名')
        plt.title('跨模型一致性最高的特征')
        plt.gca().invert_yaxis()
        
        # 6. 模型特异性特征分析
        plt.subplot(2, 3, 6)
        model_specific_counts = []
        model_names = []
        
        for model in comparison_df['model'].unique():
            model_top10 = comparison_df[comparison_df['model'] == model].nsmallest(10, 'rank')['feature'].tolist()
            other_models_top10 = set()
            
            for other_model in comparison_df['model'].unique():
                if other_model != model:
                    other_top10 = comparison_df[comparison_df['model'] == other_model].nsmallest(10, 'rank')['feature'].tolist()
                    other_models_top10.update(other_top10)
            
            # 计算该模型特有的重要特征数量
            specific_features = set(model_top10) - other_models_top10
            model_specific_counts.append(len(specific_features))
            model_names.append(model)
        
        plt.bar(model_names, model_specific_counts, color=['red', 'blue', 'green'])
        plt.xlabel('模型')
        plt.ylabel('特异性特征数量')
        plt.title('各模型特异性重要特征数量')
        
        plt.tight_layout()
        plt.savefig('../reports/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_individual_prediction_explanation(self, patient_indices=None, n_patients=5):
        """分析个体患者的预测解释"""
        if patient_indices is None:
            # 随机选择几个患者
            X_data, y_data = self.prepare_data_for_analysis()
            patient_indices = np.random.choice(len(X_data), n_patients, replace=False)
        
        explanations = {}
        
        # 为每个选定的患者生成解释
        for idx in patient_indices:
            patient_data = {}
            
            # 获取患者特征
            if 'test' in self.data:
                patient_features = self.data['test'].iloc[idx][self.feature_names]
                patient_outcome = {
                    'duration': self.data['test'].iloc[idx]['Duration'],
                    'event': self.data['test'].iloc[idx]['Event']
                }
                
                patient_data['features'] = patient_features
                patient_data['outcome'] = patient_outcome
                
                # 获取各模型的预测
                predictions = {}
                
                # 加载预测结果
                for model_name in ['deepsurv', 'cox', 'rsf']:
                    pred_file = Path(f'../data/processed/{model_name}_predictions.csv')
                    if pred_file.exists():
                        pred_data = pd.read_csv(pred_file)
                        if model_name == 'deepsurv':
                            predictions[model_name] = pred_data.iloc[idx]['Risk_Score']
                        elif model_name == 'cox':
                            predictions[model_name] = pred_data.iloc[idx]['Cox_Risk_Score']
                        else:
                            predictions[model_name] = pred_data.iloc[idx]['RSF_Risk_Score']
                
                patient_data['predictions'] = predictions
                
                # 分析特征贡献
                feature_contributions = self._analyze_feature_contributions_for_patient(patient_features, idx)
                patient_data['feature_contributions'] = feature_contributions
                
                explanations[f'Patient_{idx}'] = patient_data
        
        # 可视化个体解释
        self._plot_individual_explanations(explanations)
        
        return explanations
    
    def _analyze_feature_contributions_for_patient(self, patient_features, patient_idx):
        """分析单个患者的特征贡献"""
        contributions = {}
        
        # 对每个模型分析特征贡献
        for model_name, importance_df in self.feature_importance.items():
            if importance_df is not None:
                patient_contributions = []
                
                for _, feature_row in importance_df.head(10).iterrows():  # 只看前10重要特征
                    feature_name = feature_row['feature']
                    
                    if feature_name in patient_features.index:
                        feature_value = patient_features[feature_name]
                        
                        if model_name == 'cox':
                            # Cox模型：特征值 × 系数
                            importance = feature_row['abs_coefficient']
                            coefficient = feature_row['coefficient']
                            contribution = feature_value * coefficient
                        else:
                            # 其他模型：特征值 × 重要性
                            importance = feature_row['importance']
                            # 标准化特征值（假设特征已标准化）
                            contribution = feature_value * importance
                        
                        patient_contributions.append({
                            'feature': feature_name,
                            'value': feature_value,
                            'importance': importance,
                            'contribution': contribution
                        })
                
                contributions[model_name] = pd.DataFrame(patient_contributions)
        
        return contributions
    
    def _plot_individual_explanations(self, explanations):
        """绘制个体患者解释图"""
        n_patients = len(explanations)
        fig, axes = plt.subplots(n_patients, 2, figsize=(16, 4*n_patients))
        
        if n_patients == 1:
            axes = axes.reshape(1, -1)
        
        for i, (patient_id, patient_data) in enumerate(explanations.items()):
            # 左图：模型预测对比
            ax1 = axes[i, 0]
            
            predictions = patient_data['predictions']
            models = list(predictions.keys())
            pred_values = list(predictions.values())
            
            bars = ax1.bar(models, pred_values, color=['red', 'blue', 'green'])
            ax1.set_title(f'{patient_id} - 模型预测风险得分')
            ax1.set_ylabel('风险得分')
            
            # 添加数值标签
            for bar, value in zip(bars, pred_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 右图：主要特征贡献（以DeepSurv为例）
            ax2 = axes[i, 1]
            
            if 'deepsurv' in patient_data['feature_contributions']:
                contrib_data = patient_data['feature_contributions']['deepsurv']
                if len(contrib_data) > 0:
                    top_contrib = contrib_data.head(8)  # 前8个特征
                    
                    colors = ['red' if x < 0 else 'blue' for x in top_contrib['contribution']]
                    bars2 = ax2.barh(range(len(top_contrib)), top_contrib['contribution'], color=colors)
                    ax2.set_yticks(range(len(top_contrib)))
                    ax2.set_yticklabels(top_contrib['feature'])
                    ax2.set_xlabel('特征贡献')
                    ax2.set_title(f'{patient_id} - DeepSurv特征贡献')
                    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                    
                    # 反转y轴显示顺序
                    ax2.invert_yaxis()
            else:
                ax2.text(0.5, 0.5, '特征贡献数据不可用', transform=ax2.transAxes, 
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('../reports/individual_patient_explanations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_interpretability_report(self):
        """生成可解释性分析报告"""
        report = {
            'models_analyzed': list(self.feature_importance.keys()),
            'feature_count': len(self.feature_names),
            'analysis_methods': []
        }
        
        # 检查已完成的分析
        if 'cox' in self.feature_importance:
            report['analysis_methods'].append('Cox回归系数分析')
        
        if 'rsf' in self.feature_importance:
            report['analysis_methods'].append('随机森林特征重要性')
        
        if 'deepsurv' in self.feature_importance:
            report['analysis_methods'].append('DeepSurv代理模型分析')
        
        if 'deepsurv' in self.shap_values:
            report['analysis_methods'].append('SHAP值分析')
        
        # 生成总结
        print("=== 模型可解释性分析报告 ===")
        print(f"分析的模型数量: {len(report['models_analyzed'])}")
        print(f"特征总数: {report['feature_count']}")
        print(f"使用的分析方法: {', '.join(report['analysis_methods'])}")
        
        # 保存报告
        with open('../reports/interpretability_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("癌症生存分析模型可解释性报告\\n")
            f.write("="*50 + "\\n\\n")
            f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"分析的模型: {', '.join(report['models_analyzed'])}\\n")
            f.write(f"特征数量: {report['feature_count']}\\n")
            f.write(f"分析方法: {', '.join(report['analysis_methods'])}\\n\\n")
            
            # 添加各模型的重要特征总结
            for model_name, importance_df in self.feature_importance.items():
                if importance_df is not None:
                    f.write(f"{model_name.upper()}模型最重要的10个特征:\\n")
                    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                        f.write(f"{i+1}. {row['feature']}\\n")
                    f.write("\\n")
        
        return report

# DeepSurv包装器用于SHAP分析
class DeepSurvWrapper:
    """DeepSurv模型的包装器，用于SHAP分析"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def predict(self, X):
        """预测函数，返回风险得分"""
        if self.model is None:
            # 如果没有模型，返回随机值（用于测试）
            return np.random.randn(len(X))
        
        # 将pandas DataFrame转换为tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(X_tensor)
        
        return risk_scores.numpy().flatten()

def main():
    """主函数示例"""
    explainer = SurvivalModelExplainer()
    
    # 加载模型和数据
    success = explainer.load_models_and_data('../model', '../data/processed')
    
    if success:
        print("开始可解释性分析...")
        
        # 分析各模型
        explainer.analyze_cox_model_interpretability()
        explainer.analyze_rsf_model_interpretability()
        explainer.analyze_deepsurv_interpretability_with_shap()
        
        # 比较特征重要性
        explainer.compare_feature_importance_across_models()
        
        # 个体预测解释
        explainer.analyze_individual_prediction_explanation()
        
        # 生成报告
        explainer.generate_interpretability_report()
        
        print("可解释性分析完成！")

if __name__ == "__main__":
    main()