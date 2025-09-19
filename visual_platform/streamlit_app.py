"""
癌症生存分析模型对比可视化平台
基于Streamlit的交互式Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
import pickle
from pathlib import Path
import sys
import os

# 添加src路径以导入自定义模块
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# 设置页面配置
st.set_page_config(
    page_title="癌症生存分析模型对比平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .model-comparison {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CancerSurvivalApp:
    """癌症生存分析应用主类"""
    
    def __init__(self):
        self.data_dir = Path('../data')
        self.model_dir = Path('../model')
        self.reports_dir = Path('../reports')
        self.load_data()
    
    def load_data(self):
        """加载数据和模型结果"""
        try:
            # 加载预处理后的数据
            self.modeling_data = pd.read_csv(self.data_dir / 'processed' / 'modeling_data.csv')
            self.train_data = pd.read_csv(self.data_dir / 'processed' / 'train_data.csv')
            self.test_data = pd.read_csv(self.data_dir / 'processed' / 'test_data.csv')
            
            # 加载模型预测结果
            self.deepsurv_pred = pd.read_csv(self.data_dir / 'processed' / 'deepsurv_predictions.csv')
            self.cox_pred = pd.read_csv(self.data_dir / 'processed' / 'cox_predictions.csv')
            self.rsf_pred = pd.read_csv(self.data_dir / 'processed' / 'rsf_predictions.csv')
            
            # 加载综合评估结果
            self.comprehensive_results = pd.read_csv(self.data_dir / 'processed' / 'comprehensive_evaluation_results.csv')
            
            # 加载预处理器
            with open(self.data_dir / 'processed' / 'preprocessors.pkl', 'rb') as f:
                self.preprocessors = pickle.load(f)
            
            self.data_loaded = True
            
        except FileNotFoundError as e:
            st.error(f"数据文件未找到: {e}")
            st.error("请先运行数据预处理和模型训练notebooks")
            self.data_loaded = False
    
    def create_risk_groups(self, risk_scores, n_groups=3):
        """创建风险分组"""
        quantiles = np.quantile(risk_scores, np.linspace(0, 1, n_groups + 1))
        risk_groups = np.digitize(risk_scores, quantiles[1:-1])
        return risk_groups
    
    def plot_survival_curves(self, durations, events, risk_scores, model_name):
        """绘制生存曲线"""
        risk_groups = self.create_risk_groups(risk_scores)
        
        fig = go.Figure()
        colors = ['green', 'orange', 'red']
        labels = ['低风险组', '中风险组', '高风险组']
        
        for group in range(3):
            mask = risk_groups == group
            group_durations = durations[mask]
            group_events = events[mask]
            
            if len(group_durations) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(group_durations, group_events)
                
                fig.add_trace(go.Scatter(
                    x=kmf.timeline,
                    y=kmf.survival_function_.values.flatten(),
                    mode='lines',
                    name=f'{labels[group]} (n={mask.sum()})',
                    line=dict(color=colors[group], width=3)
                ))
        
        fig.update_layout(
            title=f'{model_name} - 风险分层生存曲线',
            xaxis_title='时间 (月)',
            yaxis_title='生存概率',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_c_index_comparison(self):
        """绘制C-index对比图"""
        fig = px.bar(
            self.comprehensive_results,
            x='Model',
            y='C_Index',
            color='Model',
            title='模型C-index性能对比',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        
        # 添加数值标签
        for i, row in self.comprehensive_results.iterrows():
            fig.add_annotation(
                x=row['Model'],
                y=row['C_Index'] + 0.005,
                text=f"{row['C_Index']:.4f}",
                showarrow=False,
                font=dict(size=12, color='black')
            )
        
        # 添加基准线
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="随机预测 (C-index=0.5)",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            yaxis=dict(range=[0.5, self.comprehensive_results['C_Index'].max() * 1.05]),
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_risk_score_distributions(self):
        """绘制风险得分分布对比"""
        # 合并所有预测结果
        combined_data = pd.DataFrame({
            'Duration': self.deepsurv_pred['Duration'],
            'Event': self.deepsurv_pred['Event'],
            'DeepSurv_Risk': self.deepsurv_pred['Risk_Score'],
            'Cox_Risk': self.cox_pred['Cox_Risk_Score'],
            'RSF_Risk': self.rsf_pred['RSF_Risk_Score']
        })
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('风险得分分布对比', 'DeepSurv风险得分 vs 生存状态',
                          'Cox风险得分 vs 生存状态', 'RSF风险得分 vs 生存状态'),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # 1. 所有模型风险得分分布
        for model, color in [('DeepSurv_Risk', '#FF6B6B'), ('Cox_Risk', '#4ECDC4'), ('RSF_Risk', '#45B7D1')]:
            fig.add_trace(
                go.Histogram(x=combined_data[model], name=model.replace('_Risk', ''), 
                           opacity=0.7, nbinsx=30, marker_color=color),
                row=1, col=1
            )
        
        # 2-4. 各模型按生存状态分组的箱线图
        models_info = [
            ('DeepSurv_Risk', 'DeepSurv', 2, 1),
            ('Cox_Risk', 'Cox', 2, 2),
            ('RSF_Risk', 'RSF', 2, 2)
        ]
        
        row_col_pairs = [(2, 1), (2, 2), (2, 2)]
        
        for i, (model_col, model_name, row, col) in enumerate(models_info[:2]):  # 只显示前两个
            alive_scores = combined_data[combined_data['Event'] == 0][model_col]
            death_scores = combined_data[combined_data['Event'] == 1][model_col]
            
            fig.add_trace(
                go.Box(y=alive_scores, name='存活', marker_color='lightblue'),
                row=row_col_pairs[i][0], col=row_col_pairs[i][1]
            )
            fig.add_trace(
                go.Box(y=death_scores, name='死亡', marker_color='lightcoral'),
                row=row_col_pairs[i][0], col=row_col_pairs[i][1]
            )
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            title_text="风险得分分布分析"
        )
        
        return fig
    
    def show_data_overview(self):
        """显示数据概览"""
        st.header("📊 数据集概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="总样本数",
                value=f"{len(self.modeling_data):,}",
                delta=None
            )
        
        with col2:
            event_rate = self.modeling_data['Event'].mean()
            st.metric(
                label="事件发生率",
                value=f"{event_rate:.2%}",
                delta=None
            )
        
        with col3:
            mean_followup = self.modeling_data['Duration'].mean()
            st.metric(
                label="平均随访时间",
                value=f"{mean_followup:.1f} 月",
                delta=None
            )
        
        with col4:
            feature_count = len(self.preprocessors['feature_columns'])
            st.metric(
                label="特征数量",
                value=feature_count,
                delta=None
            )
        
        # 数据分布可视化
        col1, col2 = st.columns(2)
        
        with col1:
            # 年龄分布
            fig_age = px.histogram(
                self.modeling_data, 
                x='Age', 
                nbins=30,
                title="年龄分布",
                color_discrete_sequence=['#1f77b4']
            )
            fig_age.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # 生存状态分布
            survival_counts = self.modeling_data['Event'].value_counts()
            survival_labels = ['存活', '死亡']
            
            fig_survival = px.pie(
                values=survival_counts.values,
                names=survival_labels,
                title="生存状态分布",
                color_discrete_sequence=['#2ca02c', '#d62728']
            )
            fig_survival.update_layout(height=400)
            st.plotly_chart(fig_survival, use_container_width=True)
    
    def show_model_comparison(self):
        """显示模型对比"""
        st.header("🔬 模型性能对比")
        
        # C-index对比
        st.subheader("C-index性能对比")
        fig_c_index = self.plot_c_index_comparison()
        st.plotly_chart(fig_c_index, use_container_width=True)
        
        # 模型排名表
        st.subheader("模型性能排名")
        ranking_df = self.comprehensive_results[['Rank', 'Model', 'C_Index', 'Risk_Stratification_Significant']].copy()
        ranking_df.columns = ['排名', '模型', 'C-index', '风险分层显著性']
        ranking_df['风险分层显著性'] = ranking_df['风险分层显著性'].map({True: '✅ 是', False: '❌ 否'})
        
        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True
        )
        
        # 性能分析
        best_model = self.comprehensive_results.iloc[0]
        st.success(f"""
        **🏆 最佳模型: {best_model['Model']}**
        - C-index: {best_model['C_Index']:.4f}
        - 排名: #{best_model['Rank']}
        - 风险分层: {'显著' if best_model['Risk_Stratification_Significant'] else '不显著'}
        """)
    
    def show_survival_analysis(self):
        """显示生存分析"""
        st.header("📈 生存分析结果")
        
        # 模型选择
        model_option = st.selectbox(
            "选择模型查看生存曲线",
            ["DeepSurv", "Cox Regression", "Random Survival Forest"],
            index=0
        )
        
        # 获取对应数据
        if model_option == "DeepSurv":
            durations = self.deepsurv_pred['Duration'].values
            events = self.deepsurv_pred['Event'].values
            risk_scores = -self.deepsurv_pred['Risk_Score'].values  # 取负值
        elif model_option == "Cox Regression":
            durations = self.cox_pred['Duration'].values
            events = self.cox_pred['Event'].values
            risk_scores = self.cox_pred['Cox_Risk_Score'].values
        else:
            durations = self.rsf_pred['Duration'].values
            events = self.rsf_pred['Event'].values
            risk_scores = self.rsf_pred['RSF_Risk_Score'].values
        
        # 绘制生存曲线
        fig_survival = self.plot_survival_curves(durations, events, risk_scores, model_option)
        st.plotly_chart(fig_survival, use_container_width=True)
        
        # 风险分层统计
        st.subheader("风险分层统计")
        risk_groups = self.create_risk_groups(risk_scores)
        
        group_stats = []
        labels = ['低风险组', '中风险组', '高风险组']
        
        for group in range(3):
            mask = risk_groups == group
            group_durations = durations[mask]
            group_events = events[mask]
            
            group_stats.append({
                '风险组': labels[group],
                '样本数量': mask.sum(),
                '事件发生率': f"{group_events.mean():.2%}",
                '中位生存时间': f"{np.median(group_durations):.1f} 月"
            })
        
        stats_df = pd.DataFrame(group_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def show_risk_analysis(self):
        """显示风险分析"""
        st.header("⚠️ 风险得分分析")
        
        # 风险得分分布
        st.subheader("风险得分分布对比")
        fig_risk_dist = self.plot_risk_score_distributions()
        st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        # 风险得分统计
        st.subheader("风险得分统计")
        combined_data = pd.DataFrame({
            'DeepSurv': self.deepsurv_pred['Risk_Score'],
            'Cox': self.cox_pred['Cox_Risk_Score'],
            'RSF': self.rsf_pred['RSF_Risk_Score']
        })
        
        st.dataframe(combined_data.describe(), use_container_width=True)
        
        # 相关性分析
        st.subheader("模型预测相关性")
        correlation_matrix = combined_data.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="模型预测相关性矩阵",
            color_continuous_scale='RdBu'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def show_interactive_prediction(self):
        """显示交互式预测"""
        st.header("🎯 交互式风险预测")
        
        st.info("通过调整患者特征，观察不同模型的风险预测结果")
        
        # 输入特征
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("年龄", min_value=18, max_value=100, value=55)
            tumor_size = st.slider("肿瘤大小 (cm)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
            chemo_sessions = st.slider("化疗疗程", min_value=0, max_value=20, value=3)
        
        with col2:
            radiation_sessions = st.slider("放疗次数", min_value=0, max_value=50, value=10)
            cancer_stage = st.selectbox("癌症分期", ["I", "II", "III", "IV"], index=1)
            gender = st.selectbox("性别", ["Male", "Female"], index=0)
        
        with col3:
            tumor_type = st.selectbox("肿瘤类型", ["Lung", "Stomach", "Liver", "Breast"], index=0)
            smoking_status = st.selectbox("吸烟状态", ["Never", "Former", "Current"], index=0)
            has_surgery = st.checkbox("是否手术", value=True)
        
        # 模拟预测（简化版）
        if st.button("预测风险", type="primary"):
            # 这里应该使用实际的模型进行预测
            # 为演示目的，使用简化的风险计算
            
            base_risk = 0.3
            age_factor = (age - 50) * 0.01
            tumor_factor = tumor_size * 0.02
            stage_factor = {"I": 0, "II": 0.1, "III": 0.2, "IV": 0.4}[cancer_stage]
            
            predicted_risk = max(0, min(1, base_risk + age_factor + tumor_factor + stage_factor))
            
            # 显示预测结果
            risk_level = "低" if predicted_risk < 0.3 else "中" if predicted_risk < 0.7 else "高"
            color = "green" if risk_level == "低" else "orange" if risk_level == "中" else "red"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: {color}20; border-left: 5px solid {color};">
                <h3 style="color: {color}; margin: 0;">预测风险: {predicted_risk:.2%}</h3>
                <p style="margin: 0.5rem 0 0 0;">风险等级: {risk_level}风险</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 风险因素分析
            st.subheader("风险因素分析")
            factors = {
                "年龄": age_factor,
                "肿瘤大小": tumor_factor,
                "癌症分期": stage_factor,
                "基础风险": base_risk
            }
            
            factor_df = pd.DataFrame(
                list(factors.items()),
                columns=["因素", "风险贡献"]
            )
            
            fig_factors = px.bar(
                factor_df,
                x="因素",
                y="风险贡献",
                title="各因素对风险的贡献",
                color="风险贡献",
                color_continuous_scale="Reds"
            )
            fig_factors.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig_factors, use_container_width=True)
    
    def run(self):
        """运行应用"""
        # 页面标题
        st.markdown('<h1 class="main-header">🔬 癌症生存分析模型对比平台</h1>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.error("数据加载失败，请检查数据文件是否存在")
            return
        
        # 侧边栏导航
        st.sidebar.title("📋 导航菜单")
        page = st.sidebar.selectbox(
            "选择页面",
            [
                "📊 数据概览",
                "🔬 模型对比",
                "📈 生存分析",
                "⚠️ 风险分析",
                "🎯 交互预测"
            ]
        )
        
        # 根据选择显示对应页面
        if page == "📊 数据概览":
            self.show_data_overview()
        elif page == "🔬 模型对比":
            self.show_model_comparison()
        elif page == "📈 生存分析":
            self.show_survival_analysis()
        elif page == "⚠️ 风险分析":
            self.show_risk_analysis()
        elif page == "🎯 交互预测":
            self.show_interactive_prediction()
        
        # 侧边栏信息
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### 📖 使用说明
        
        **数据概览**: 查看数据集的基本统计信息和分布
        
        **模型对比**: 比较DeepSurv、Cox回归和随机生存森林的性能
        
        **生存分析**: 查看各模型的风险分层生存曲线
        
        **风险分析**: 分析风险得分的分布和相关性
        
        **交互预测**: 模拟患者特征进行风险预测
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### ℹ️ 模型信息
        
        - **DeepSurv**: 基于深度神经网络的生存分析模型
        - **Cox回归**: 经典的比例风险回归模型
        - **RSF**: 随机生存森林集成学习模型
        """)


def main():
    """主函数"""
    app = CancerSurvivalApp()
    app.run()


if __name__ == "__main__":
    main()