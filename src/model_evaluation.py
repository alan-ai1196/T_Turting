"""
模型评估工具模块
提供生存分析模型的综合评估功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from scipy import stats
import pickle
from pathlib import Path


class SurvivalModelEvaluator:
    """生存分析模型评估器"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
    
    def load_predictions(self, predictions_dir):
        """加载模型预测结果"""
        predictions_dir = Path(predictions_dir)
        
        # 加载DeepSurv预测
        if (predictions_dir / 'deepsurv_predictions.csv').exists():
            deepsurv_pred = pd.read_csv(predictions_dir / 'deepsurv_predictions.csv')
            self.predictions['deepsurv'] = deepsurv_pred
        
        # 加载Cox预测
        if (predictions_dir / 'cox_predictions.csv').exists():
            cox_pred = pd.read_csv(predictions_dir / 'cox_predictions.csv')
            self.predictions['cox'] = cox_pred
        
        # 加载RSF预测
        if (predictions_dir / 'rsf_predictions.csv').exists():
            rsf_pred = pd.read_csv(predictions_dir / 'rsf_predictions.csv')
            self.predictions['rsf'] = rsf_pred
        
        print(f"已加载 {len(self.predictions)} 个模型的预测结果")
    
    def calculate_c_indices(self):
        """计算所有模型的C-index"""
        c_indices = {}
        
        # 确保所有预测结果有相同的数据
        base_data = None
        for model_name in self.predictions:
            if base_data is None:
                base_data = self.predictions[model_name][['Duration', 'Event']].copy()
            
        durations = base_data['Duration'].values
        events = base_data['Event'].values
        
        # 计算各模型的C-index
        if 'deepsurv' in self.predictions:
            risk_scores = self.predictions['deepsurv']['Risk_Score'].values
            c_indices['DeepSurv'] = concordance_index(durations, -risk_scores, events)
        
        if 'cox' in self.predictions:
            risk_scores = self.predictions['cox']['Cox_Risk_Score'].values
            c_indices['Cox Regression'] = concordance_index(durations, risk_scores, events)
        
        if 'rsf' in self.predictions:
            risk_scores = self.predictions['rsf']['RSF_Risk_Score'].values
            c_indices['Random Survival Forest'] = concordance_index_censored(
                events.astype(bool), durations, risk_scores)[0]
        
        self.results['c_indices'] = c_indices
        return c_indices
    
    def create_risk_groups(self, risk_scores, n_groups=3):
        """创建风险分组"""
        quantiles = np.quantile(risk_scores, np.linspace(0, 1, n_groups + 1))
        risk_groups = np.digitize(risk_scores, quantiles[1:-1])
        return risk_groups
    
    def evaluate_risk_stratification(self):
        """评估风险分层能力"""
        base_data = None
        for model_name in self.predictions:
            if base_data is None:
                base_data = self.predictions[model_name][['Duration', 'Event']].copy()
        
        durations = base_data['Duration'].values
        events = base_data['Event'].values
        
        stratification_results = {}
        
        # 评估各模型的风险分层
        for model_key, model_name in [('deepsurv', 'DeepSurv'), 
                                     ('cox', 'Cox Regression'), 
                                     ('rsf', 'Random Survival Forest')]:
            if model_key in self.predictions:
                if model_key == 'deepsurv':
                    risk_scores = -self.predictions[model_key]['Risk_Score'].values  # DeepSurv需要取负值
                elif model_key == 'cox':
                    risk_scores = self.predictions[model_key]['Cox_Risk_Score'].values
                else:
                    risk_scores = self.predictions[model_key]['RSF_Risk_Score'].values
                
                # 计算log-rank test
                logrank_p = self.calculate_logrank_test(durations, events, risk_scores)
                
                # 计算风险组统计
                risk_groups = self.create_risk_groups(risk_scores)
                group_stats = self.calculate_risk_group_stats(durations, events, risk_groups)
                
                stratification_results[model_name] = {
                    'logrank_p_value': logrank_p,
                    'significant': logrank_p < 0.05,
                    'group_stats': group_stats
                }
        
        self.results['risk_stratification'] = stratification_results
        return stratification_results
    
    def calculate_logrank_test(self, durations, events, risk_scores):
        """计算风险分层的log-rank检验"""
        risk_groups = self.create_risk_groups(risk_scores)
        
        # 低风险组 vs 高风险组
        low_risk_mask = risk_groups == 0
        high_risk_mask = risk_groups == 2
        
        low_risk_durations = durations[low_risk_mask]
        low_risk_events = events[low_risk_mask]
        high_risk_durations = durations[high_risk_mask]
        high_risk_events = events[high_risk_mask]
        
        results = logrank_test(low_risk_durations, high_risk_durations, 
                              low_risk_events, high_risk_events)
        
        return results.p_value
    
    def calculate_risk_group_stats(self, durations, events, risk_groups):
        """计算风险组统计信息"""
        group_stats = []
        labels = ['低风险组', '中风险组', '高风险组']
        
        for group in range(3):
            mask = risk_groups == group
            group_durations = durations[mask]
            group_events = events[mask]
            
            stats = {
                'group': labels[group],
                'sample_count': mask.sum(),
                'event_rate': group_events.mean(),
                'median_duration': np.median(group_durations)
            }
            group_stats.append(stats)
        
        return pd.DataFrame(group_stats)
    
    def plot_survival_curves_comparison(self, save_path=None):
        """绘制所有模型的风险分层生存曲线对比"""
        base_data = None
        for model_name in self.predictions:
            if base_data is None:
                base_data = self.predictions[model_name][['Duration', 'Event']].copy()
        
        durations = base_data['Duration'].values
        events = base_data['Event'].values
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        model_info = [
            ('deepsurv', 'DeepSurv', 'Risk_Score', True),  # True表示需要取负值
            ('cox', 'Cox Regression', 'Cox_Risk_Score', False),
            ('rsf', 'Random Survival Forest', 'RSF_Risk_Score', False)
        ]
        
        ax_idx = 0
        for model_key, model_name, score_col, need_negative in model_info:
            if model_key in self.predictions:
                risk_scores = self.predictions[model_key][score_col].values
                if need_negative:
                    risk_scores = -risk_scores
                
                self.plot_single_model_survival_curves(
                    durations, events, risk_scores, model_name, axes[ax_idx]
                )
                ax_idx += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_single_model_survival_curves(self, durations, events, risk_scores, model_name, ax):
        """绘制单个模型的风险分层生存曲线"""
        risk_groups = self.create_risk_groups(risk_scores)
        
        kmf = KaplanMeierFitter()
        colors = ['green', 'orange', 'red']
        labels = ['低风险组', '中风险组', '高风险组']
        
        for group in range(3):
            mask = risk_groups == group
            group_durations = durations[mask]
            group_events = events[mask]
            
            kmf.fit(group_durations, group_events, label=f'{labels[group]} (n={mask.sum()})')
            kmf.plot_survival_function(ax=ax, color=colors[group], linewidth=2)
        
        ax.set_title(f'{model_name} - 风险分层生存曲线', fontsize=12)
        ax.set_xlabel('时间 (月)', fontsize=10)
        ax.set_ylabel('生存概率', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def plot_c_index_comparison(self, save_path=None):
        """绘制C-index对比图"""
        c_indices = self.results.get('c_indices', self.calculate_c_indices())
        
        models = list(c_indices.keys())
        values = list(c_indices.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(models)]
        
        bars = plt.bar(models, values, color=colors, alpha=0.8)
        
        plt.title('生存分析模型C-index性能比较', fontsize=16, pad=20)
        plt.ylabel('C-index', fontsize=12)
        plt.xlabel('模型', fontsize=12)
        plt.ylim(0.5, max(values) * 1.05)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 添加基准线
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='随机预测 (C-index=0.5)')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_risk_score_discrimination(self):
        """计算风险得分在不同生存状态下的区分能力"""
        base_data = None
        for model_name in self.predictions:
            if base_data is None:
                base_data = self.predictions[model_name][['Duration', 'Event']].copy()
        
        events = base_data['Event'].values
        discrimination_results = {}
        
        for model_key, model_name in [('deepsurv', 'DeepSurv'), 
                                     ('cox', 'Cox Regression'), 
                                     ('rsf', 'Random Survival Forest')]:
            if model_key in self.predictions:
                if model_key == 'deepsurv':
                    risk_scores = self.predictions[model_key]['Risk_Score'].values
                elif model_key == 'cox':
                    risk_scores = self.predictions[model_key]['Cox_Risk_Score'].values
                else:
                    risk_scores = self.predictions[model_key]['RSF_Risk_Score'].values
                
                # 计算统计差异
                alive_scores = risk_scores[events == 0]
                death_scores = risk_scores[events == 1]
                
                t_stat, p_value = stats.ttest_ind(death_scores, alive_scores)
                
                discrimination_results[model_name] = {
                    'alive_mean': alive_scores.mean(),
                    'alive_std': alive_scores.std(),
                    'death_mean': death_scores.mean(),
                    'death_std': death_scores.std(),
                    't_statistic': t_stat,
                    'p_value': p_value
                }
        
        self.results['risk_score_discrimination'] = discrimination_results
        return discrimination_results
    
    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        # 确保所有评估都已完成
        c_indices = self.results.get('c_indices', self.calculate_c_indices())
        risk_stratification = self.results.get('risk_stratification', self.evaluate_risk_stratification())
        discrimination = self.results.get('risk_score_discrimination', self.calculate_risk_score_discrimination())
        
        # 创建综合结果DataFrame
        models = list(c_indices.keys())
        comprehensive_results = pd.DataFrame({
            'Model': models,
            'C_Index': [c_indices[model] for model in models],
            'LogRank_P_Value': [risk_stratification[model]['logrank_p_value'] for model in models],
            'Risk_Stratification_Significant': [risk_stratification[model]['significant'] for model in models],
            'Risk_Score_Discrimination_P': [discrimination[model]['p_value'] for model in models]
        })
        
        # 按C-index排序
        comprehensive_results = comprehensive_results.sort_values('C_Index', ascending=False)
        comprehensive_results['Rank'] = range(1, len(comprehensive_results) + 1)
        
        self.results['comprehensive'] = comprehensive_results
        return comprehensive_results
    
    def save_results(self, save_dir):
        """保存评估结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 保存综合结果
        if 'comprehensive' in self.results:
            self.results['comprehensive'].to_csv(save_dir / 'comprehensive_evaluation_results.csv', index=False)
        
        # 保存C-index结果
        if 'c_indices' in self.results:
            c_index_df = pd.DataFrame(list(self.results['c_indices'].items()), 
                                    columns=['Model', 'C_Index'])
            c_index_df.to_csv(save_dir / 'c_index_results.csv', index=False)
        
        # 保存风险分层结果
        if 'risk_stratification' in self.results:
            stratification_data = []
            for model, data in self.results['risk_stratification'].items():
                stratification_data.append({
                    'Model': model,
                    'LogRank_P_Value': data['logrank_p_value'],
                    'Significant': data['significant']
                })
            stratification_df = pd.DataFrame(stratification_data)
            stratification_df.to_csv(save_dir / 'risk_stratification_results.csv', index=False)
        
        # 保存完整结果
        with open(save_dir / 'evaluation_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"评估结果已保存至: {save_dir}")
    
    def plot_comprehensive_evaluation(self, save_path=None):
        """绘制综合评估图表"""
        if 'comprehensive' not in self.results:
            self.generate_comprehensive_report()
        
        comprehensive_results = self.results['comprehensive']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. C-index对比
        models = comprehensive_results['Model']
        c_indices = comprehensive_results['C_Index']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(models)]
        
        bars = ax1.bar(models, c_indices, color=colors, alpha=0.8)
        ax1.set_title('C-index性能对比', fontsize=14)
        ax1.set_ylabel('C-index')
        ax1.set_ylim(0.5, max(c_indices) * 1.05)
        
        for bar, value in zip(bars, c_indices):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 风险分层显著性
        significance = [-np.log10(p) for p in comprehensive_results['LogRank_P_Value']]
        bars2 = ax2.bar(models, significance, color=['#FFB6C1', '#98FB98', '#87CEEB'], alpha=0.8)
        ax2.set_title('风险分层显著性 (-log10(p-value))', fontsize=14)
        ax2.set_ylabel('-log10(p-value)')
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax2.legend()
        
        for bar, p_val in zip(bars2, comprehensive_results['LogRank_P_Value']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'p={p_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 模型复杂度 vs 性能
        complexity_scores = [3, 1, 2][:len(models)]  # DeepSurv最复杂，Cox最简单，RSF中等
        
        ax3.scatter(complexity_scores, c_indices, s=200, alpha=0.7, c=colors)
        for i, model in enumerate(models):
            ax3.annotate(model, (complexity_scores[i], c_indices[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('模型复杂度 (1=简单, 3=复杂)')
        ax3.set_ylabel('C-index')
        ax3.set_title('模型复杂度 vs 性能')
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能排名
        ranking_colors = ['gold', 'silver', '#CD7F32'][:len(models)]
        
        bars4 = ax4.barh(models, c_indices, color=ranking_colors, alpha=0.8)
        ax4.set_xlabel('C-index')
        ax4.set_title('模型性能排名')
        
        for i, (bar, rank) in enumerate(zip(bars4, comprehensive_results['Rank'])):
            ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f'#{rank}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数示例"""
    # 创建评估器
    evaluator = SurvivalModelEvaluator()
    
    # 加载预测结果
    evaluator.load_predictions('../data/processed')
    
    # 进行各项评估
    c_indices = evaluator.calculate_c_indices()
    print("C-index结果:")
    for model, c_index in c_indices.items():
        print(f"{model}: {c_index:.4f}")
    
    # 风险分层评估
    risk_stratification = evaluator.evaluate_risk_stratification()
    print("\\n风险分层评估:")
    for model, results in risk_stratification.items():
        print(f"{model}: p-value = {results['logrank_p_value']:.4f}, "
              f"Significant = {results['significant']}")
    
    # 生成综合报告
    comprehensive_results = evaluator.generate_comprehensive_report()
    print("\\n综合评估结果:")
    print(comprehensive_results)
    
    # 绘制图表
    evaluator.plot_c_index_comparison('../reports/c_index_comparison.png')
    evaluator.plot_survival_curves_comparison('../reports/survival_curves_comparison.png')
    evaluator.plot_comprehensive_evaluation('../reports/comprehensive_evaluation.png')
    
    # 保存结果
    evaluator.save_results('../data/processed')
    
    print("\\n模型评估完成！")


if __name__ == "__main__":
    main()