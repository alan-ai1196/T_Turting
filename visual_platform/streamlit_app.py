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

# 导入预测服务
try:
    from prediction_service import DeepSurvPredictor, PatientDataValidator
except ImportError:
    st.error("无法导入预测服务模块，请检查src/prediction_service.py文件")
    DeepSurvPredictor = None
    PatientDataValidator = None

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
        
        # 初始化预测器
        self.predictor = None
        if DeepSurvPredictor is not None:
            self.predictor = DeepSurvPredictor()
        
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
        
        # 创建多个标签页
        tab1, tab2, tab3, tab4 = st.tabs(["C-index对比", "Brier Score分析", "综合评估", "性能排名"])
        
        with tab1:
            # C-index对比
            st.subheader("C-index性能对比")
            fig_c_index = self.plot_c_index_comparison()
            st.plotly_chart(fig_c_index, use_container_width=True)
            
            # C-index解释
            st.info("""
            **C-index (一致性指数)** 衡量模型预测排序与实际生存时间排序的一致性。
            - 范围: 0.5-1.0，越高越好
            - > 0.7: 优秀性能
            - 0.6-0.7: 良好性能
            - < 0.6: 一般性能
            """)
        
        with tab2:
            # Brier Score分析
            st.subheader("Brier Score 和 集成Brier Score (IBS)")
            
            # 尝试加载Brier Score数据
            try:
                brier_file = self.data_dir / 'processed' / 'brier_scores_results.csv'
                ibs_file = self.data_dir / 'processed' / 'integrated_brier_scores_results.csv'
                
                if brier_file.exists():
                    brier_data = pd.read_csv(brier_file)
                    
                    # Brier Score随时间变化
                    fig_brier = px.line(
                        brier_data, 
                        x='Time_Point', 
                        y='Brier_Score', 
                        color='Model',
                        title='Brier Score随时间变化',
                        markers=True
                    )
                    fig_brier.update_layout(
                        xaxis_title='时间 (月)',
                        yaxis_title='Brier Score',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_brier, use_container_width=True)
                    
                    # 平均Brier Score对比
                    avg_brier = brier_data.groupby('Model')['Brier_Score'].mean().reset_index()
                    fig_avg_brier = px.bar(
                        avg_brier,
                        x='Model',
                        y='Brier_Score',
                        title='平均Brier Score对比 (越低越好)',
                        color='Model'
                    )
                    st.plotly_chart(fig_avg_brier, use_container_width=True)
                
                if ibs_file.exists():
                    ibs_data = pd.read_csv(ibs_file)
                    
                    # IBS对比
                    fig_ibs = px.bar(
                        ibs_data,
                        x='Model',
                        y='IBS',
                        title='集成Brier Score (IBS) 对比 (越低越好)',
                        color='Model'
                    )
                    st.plotly_chart(fig_ibs, use_container_width=True)
                    
                    # 显示IBS表格
                    st.subheader("IBS详细结果")
                    display_ibs = ibs_data[['Model', 'IBS', 'Time_Range_Start', 'Time_Range_End']].copy()
                    display_ibs.columns = ['模型', 'IBS', '时间范围开始', '时间范围结束']
                    st.dataframe(display_ibs, use_container_width=True, hide_index=True)
                
                if not brier_file.exists() and not ibs_file.exists():
                    st.warning("Brier Score和IBS数据尚未生成，请先运行增强版模型评估notebook")
                    
            except Exception as e:
                st.error(f"加载Brier Score数据时出错: {e}")
            
            # Brier Score解释
            st.info("""
            **Brier Score** 衡量预测概率与实际结果的平方差，是时间依赖的准确性指标。
            
            **集成Brier Score (IBS)** 是Brier Score在整个时间范围内的积分，提供综合性能评估。
            
            两个指标都是越低越好，表明预测越准确。
            """)
        
        with tab3:
            # 综合评估
            st.subheader("综合性能评估")
            
            # 检查是否有新的评估指标
            if 'Mean_Brier_Score' in self.comprehensive_results.columns:
                # 显示包含所有指标的综合表格
                display_cols = ['Rank', 'Model', 'C_Index', 'Mean_Brier_Score', 'Integrated_Brier_Score', 
                               'Risk_Stratification_Significant']
                if 'Performance_Grade' in self.comprehensive_results.columns:
                    display_cols.append('Performance_Grade')
                
                enhanced_df = self.comprehensive_results[display_cols].copy()
                col_names = ['排名', '模型', 'C-index', '平均Brier Score', 'IBS', '风险分层显著性']
                if 'Performance_Grade' in self.comprehensive_results.columns:
                    col_names.append('性能等级')
                
                enhanced_df.columns = col_names
                enhanced_df['风险分层显著性'] = enhanced_df['风险分层显著性'].map({True: '✅ 是', False: '❌ 否'})
                
                st.dataframe(enhanced_df, use_container_width=True, hide_index=True)
                
                # 性能雷达图
                if len(self.comprehensive_results) >= 2:
                    self._plot_performance_radar()
            else:
                # 显示基础评估表格
                ranking_df = self.comprehensive_results[['Rank', 'Model', 'C_Index', 'Risk_Stratification_Significant']].copy()
                ranking_df.columns = ['排名', '模型', 'C-index', '风险分层显著性']
                ranking_df['风险分层显著性'] = ranking_df['风险分层显著性'].map({True: '✅ 是', False: '❌ 否'})
                
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        
        with tab4:
            # 性能排名和最佳模型分析
            st.subheader("🏆 最佳模型分析")
            
            best_model = self.comprehensive_results.iloc[0]
            
            # 创建指标卡片
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="最佳模型",
                    value=best_model['Model'],
                    delta=f"排名 #{best_model['Rank']}"
                )
            
            with col2:
                st.metric(
                    label="C-index",
                    value=f"{best_model['C_Index']:.4f}",
                    delta="越高越好"
                )
            
            with col3:
                risk_strat = "显著" if best_model['Risk_Stratification_Significant'] else "不显著"
                st.metric(
                    label="风险分层",
                    value=risk_strat,
                    delta="统计检验结果"
                )
            
            # 模型优势分析
            st.subheader("📊 模型对比分析")
            
            if len(self.comprehensive_results) > 1:
                # C-index差异分析
                c_index_diff = self.comprehensive_results['C_Index'].max() - self.comprehensive_results['C_Index'].min()
                
                if c_index_diff > 0.05:
                    st.success(f"🎯 模型间存在显著性能差异 (最大差异: {c_index_diff:.4f})")
                else:
                    st.info(f"📈 模型间性能较为接近 (最大差异: {c_index_diff:.4f})")
                
                # 显示每个模型的优势
                st.write("**各模型特点:**")
                for _, model_row in self.comprehensive_results.iterrows():
                    model_name = model_row['Model']
                    c_index = model_row['C_Index']
                    
                    if model_name == "DeepSurv":
                        advantages = "深度学习，非线性建模，自动特征交互"
                    elif model_name == "Cox Regression":
                        advantages = "经典方法，可解释性强，计算高效"
                    else:
                        advantages = "集成学习，处理非线性，特征重要性"
                    
                    st.write(f"- **{model_name}** (C-index: {c_index:.4f}): {advantages}")
    
    def _plot_performance_radar(self):
        """绘制性能雷达图"""
        try:
            # 准备雷达图数据
            models = self.comprehensive_results['Model'].tolist()
            
            # 标准化指标（C-index保持原值，Brier Score和IBS取倒数并标准化）
            metrics = []
            metric_names = []
            
            if 'C_Index' in self.comprehensive_results.columns:
                metrics.append(self.comprehensive_results['C_Index'].tolist())
                metric_names.append('C-index')
            
            if 'Mean_Brier_Score' in self.comprehensive_results.columns:
                # Brier Score越低越好，所以用1减去标准化值
                brier_scores = self.comprehensive_results['Mean_Brier_Score'].fillna(0.5)
                normalized_brier = 1 - (brier_scores - brier_scores.min()) / (brier_scores.max() - brier_scores.min() + 1e-8)
                metrics.append(normalized_brier.tolist())
                metric_names.append('Brier Score (标准化)')
            
            if 'Integrated_Brier_Score' in self.comprehensive_results.columns:
                ibs_scores = self.comprehensive_results['Integrated_Brier_Score'].fillna(0.5)
                normalized_ibs = 1 - (ibs_scores - ibs_scores.min()) / (ibs_scores.max() - ibs_scores.min() + 1e-8)
                metrics.append(normalized_ibs.tolist())
                metric_names.append('IBS (标准化)')
            
            if len(metrics) >= 2:
                # 创建雷达图
                fig = go.Figure()
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                for i, model in enumerate(models):
                    model_values = [metric[i] for metric in metrics]
                    model_values.append(model_values[0])  # 闭合图形
                    
                    fig.add_trace(go.Scatterpolar(
                        r=model_values,
                        theta=metric_names + [metric_names[0]],
                        fill='toself',
                        name=model,
                        line_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="模型综合性能雷达图",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"绘制雷达图时出错: {e}")
    
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
        """显示增强的交互式预测"""
        st.header("🎯 智能风险预测系统")
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4 style="margin: 0; color: #1f77b4;">🏥 患者风险评估工具</h4>
        <p style="margin: 0.5rem 0 0 0;">基于DeepSurv深度学习模型，提供个性化的癌症生存风险预测</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建两列布局
        col_input, col_results = st.columns([1, 1])
        
        with col_input:
            st.subheader("📝 患者信息输入")
            
            # 基本信息
            with st.expander("👤 基本信息", expanded=True):
                age = st.slider("年龄", min_value=18, max_value=100, value=55, 
                              help="患者年龄，年龄越大通常风险越高")
                
                # 性别选择（如果需要）
                gender_options = {"女性": 0, "男性": 1}
                gender = st.selectbox("性别", list(gender_options.keys()), index=0)
                gender_value = gender_options[gender]
            
            # 肿瘤特征
            with st.expander("🔬 肿瘤特征", expanded=True):
                tumor_size = st.slider("肿瘤大小 (cm)", min_value=0.1, max_value=20.0, 
                                     value=3.0, step=0.1,
                                     help="原发肿瘤的最大直径")
                
                stage_options = {"I期": 1, "II期": 2, "III期": 3, "IV期": 4}
                stage_label = st.selectbox("癌症分期", list(stage_options.keys()), index=1)
                stage = stage_options[stage_label]
                
                grade_options = {"低分化": 1, "中分化": 2, "高分化": 3}
                grade_label = st.selectbox("肿瘤分级", list(grade_options.keys()), index=1)
                grade = grade_options[grade_label]
                
                lymph_nodes = st.number_input("阳性淋巴结数量", min_value=0, max_value=50, 
                                            value=0, step=1,
                                            help="检测到癌细胞的淋巴结数量")
            
            # 分子标记
            with st.expander("🧬 分子标记", expanded=True):
                er_positive = st.checkbox("ER阳性 (雌激素受体)", value=True,
                                        help="雌激素受体阳性通常预后较好")
                pr_positive = st.checkbox("PR阳性 (孕激素受体)", value=True,
                                        help="孕激素受体阳性通常预后较好")
                her2_positive = st.checkbox("HER2阳性", value=False,
                                          help="HER2阳性可能需要靶向治疗")
            
            # 治疗方案
            with st.expander("💊 治疗方案", expanded=True):
                surgery_options = {"无手术": 0, "保乳手术": 1, "乳房切除术": 2, "根治性手术": 3}
                surgery_label = st.selectbox("手术类型", list(surgery_options.keys()), index=2)
                surgery_type = surgery_options[surgery_label]
                
                chemotherapy = st.checkbox("化疗", value=True,
                                         help="是否接受化疗治疗")
                radiotherapy = st.checkbox("放疗", value=True,
                                         help="是否接受放射治疗")
            
            # 其他因素
            with st.expander("📋 其他因素"):
                menopause_options = {"绝经前": 0, "围绝经期": 1, "绝经后": 2}
                menopause_label = st.selectbox("绝经状态", list(menopause_options.keys()), index=0)
                menopause_status = menopause_options[menopause_label]
                
                histology_options = {"导管癌": 0, "小叶癌": 1, "混合型": 2, "其他": 3}
                histology_label = st.selectbox("病理类型", list(histology_options.keys()), index=0)
                histology_type = histology_options[histology_label]
        
        with col_results:
            st.subheader("📊 预测结果")
            
            # 创建患者数据字典
            patient_data = {
                'age': age,
                'tumor_size': tumor_size,
                'stage': stage,
                'grade': grade,
                'lymph_nodes': lymph_nodes,
                'er_positive': int(er_positive),
                'pr_positive': int(pr_positive),
                'her2_positive': int(her2_positive),
                'surgery_type': surgery_type,
                'chemotherapy': int(chemotherapy),
                'radiotherapy': int(radiotherapy),
                'menopause_status': menopause_status,
                'histology_type': histology_type
            }
            
            # 验证数据
            if PatientDataValidator is not None:
                is_valid, errors = PatientDataValidator.validate_patient_data(patient_data)
                if not is_valid:
                    st.error("输入数据有误：")
                    for error in errors:
                        st.error(f"• {error}")
                    return
            
            # 进行预测
            if st.button("🔮 开始预测", type="primary", use_container_width=True):
                with st.spinner("正在分析患者数据..."):
                    
                    # 获取风险评分
                    if self.predictor is not None:
                        risk_score = self.predictor.predict_risk_score(patient_data)
                        risk_interpretation = self.predictor.get_risk_interpretation(risk_score)
                        feature_importance = self.predictor.get_feature_importance(patient_data)
                    else:
                        # 使用简化的风险计算
                        risk_score = self._calculate_simple_risk(patient_data)
                        risk_interpretation = self._get_simple_risk_interpretation(risk_score)
                        feature_importance = self._get_simple_feature_importance(patient_data)
                    
                    # 显示风险评分
                    self._display_risk_score(risk_score, risk_interpretation)
                    
                    # 显示生存概率曲线
                    self._display_survival_curve(patient_data, risk_score)
                    
                    # 显示特征重要性
                    self._display_feature_importance(feature_importance)
                    
                    # 显示建议
                    self._display_recommendations(risk_interpretation)
    
    def _calculate_simple_risk(self, patient_data):
        """简化的风险计算"""
        base_risk = 0.3
        
        # 年龄因子
        age_factor = (patient_data['age'] - 50) * 0.008
        
        # 肿瘤大小因子
        tumor_factor = patient_data['tumor_size'] * 0.02
        
        # 分期因子
        stage_factor = (patient_data['stage'] - 1) * 0.12
        
        # 淋巴结因子
        lymph_factor = patient_data['lymph_nodes'] * 0.015
        
        # 分级因子
        grade_factor = (patient_data['grade'] - 1) * 0.06
        
        # 受体状态因子
        er_factor = -0.08 if patient_data['er_positive'] else 0.06
        pr_factor = -0.05 if patient_data['pr_positive'] else 0.03
        her2_factor = 0.05 if patient_data['her2_positive'] else 0
        
        # 治疗因子
        chemo_factor = -0.10 if patient_data['chemotherapy'] else 0.05
        radio_factor = -0.06 if patient_data['radiotherapy'] else 0.03
        surgery_factor = -patient_data['surgery_type'] * 0.03
        
        total_risk = (base_risk + age_factor + tumor_factor + stage_factor + 
                     lymph_factor + grade_factor + er_factor + pr_factor + 
                     her2_factor + chemo_factor + radio_factor + surgery_factor)
        
        return max(0.0, min(1.0, total_risk))
    
    def _get_simple_risk_interpretation(self, risk_score):
        """简化的风险解释"""
        if risk_score < 0.3:
            return {
                'risk_level': '低风险',
                'color': 'green',
                'description': '患者的预测风险较低，预后相对良好。',
                'recommendations': [
                    '定期随访观察',
                    '保持健康生活方式',
                    '按医嘱进行常规检查'
                ]
            }
        elif risk_score < 0.7:
            return {
                'risk_level': '中等风险',
                'color': 'orange',
                'description': '患者的预测风险处于中等水平，需要密切关注。',
                'recommendations': [
                    '加强定期监测',
                    '考虑辅助治疗',
                    '保持良好的生活习惯',
                    '心理支持和指导'
                ]
            }
        else:
            return {
                'risk_level': '高风险',
                'color': 'red',
                'description': '患者的预测风险较高，需要积极的治疗和监护。',
                'recommendations': [
                    '制定积极的治疗方案',
                    '频繁的医学监测',
                    '考虑多学科会诊',
                    '提供心理支持',
                    '家属参与护理决策'
                ]
            }
    
    def _get_simple_feature_importance(self, patient_data):
        """简化的特征重要性计算"""
        return {
            'tumor_size': patient_data['tumor_size'] * 0.02,
            'stage': (patient_data['stage'] - 1) * 0.12,
            'lymph_nodes': patient_data['lymph_nodes'] * 0.015,
            'age': abs(patient_data['age'] - 50) * 0.008,
            'grade': (patient_data['grade'] - 1) * 0.06
        }
    
    def _display_risk_score(self, risk_score, risk_interpretation):
        """显示风险评分"""
        color = risk_interpretation['color']
        risk_level = risk_interpretation['risk_level']
        description = risk_interpretation['description']
        
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {color}20; 
                    border-left: 5px solid {color}; margin-bottom: 1rem;">
            <h3 style="color: {color}; margin: 0;">🎯 风险评分: {risk_score:.1%}</h3>
            <h4 style="color: {color}; margin: 0.5rem 0;">等级: {risk_level}</h4>
            <p style="margin: 0.5rem 0 0 0; color: #333;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 风险计量器
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "风险指数 (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    def _display_survival_curve(self, patient_data, risk_score):
        """显示生存概率曲线"""
        # 生成时间点
        time_points = np.linspace(0, 120, 100)  # 0-120个月
        
        # 计算生存概率
        if self.predictor is not None:
            survival_probs = self.predictor.predict_survival_probability(
                patient_data, time_points
            )
        else:
            # 简化的生存概率计算
            decay_rate = risk_score * 0.08
            survival_probs = [np.exp(-decay_rate * t / 12) for t in time_points]
        
        # 创建生存曲线图
        fig_survival = go.Figure()
        
        fig_survival.add_trace(go.Scatter(
            x=time_points,
            y=survival_probs,
            mode='lines',
            name='预测生存概率',
            line=dict(color='blue', width=3),
            fill='tonexty',
            fillcolor='rgba(0,100,255,0.1)'
        ))
        
        # 添加关键时间点标记
        key_times = [12, 24, 36, 60]  # 1年、2年、3年、5年
        for t in key_times:
            if t <= max(time_points):
                idx = np.argmin(np.abs(np.array(time_points) - t))
                prob = survival_probs[idx]
                fig_survival.add_annotation(
                    x=t, y=prob,
                    text=f"{int(t/12)}年: {prob:.1%}",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="white",
                    bordercolor="blue"
                )
        
        fig_survival.update_layout(
            title="个体化生存概率预测曲线",
            xaxis_title="时间 (月)",
            yaxis_title="生存概率",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_survival, use_container_width=True)
    
    def _display_feature_importance(self, feature_importance):
        """显示特征重要性"""
        if not feature_importance:
            return
        
        st.subheader("📈 风险因子分析")
        
        # 创建特征重要性数据框
        feature_df = pd.DataFrame([
            {'特征': feature, '重要性': importance}
            for feature, importance in feature_importance.items()
        ]).sort_values('重要性', ascending=True)
        
        # 特征名称映射
        feature_names = {
            'tumor_size': '肿瘤大小',
            'stage': '癌症分期',
            'lymph_nodes': '淋巴结',
            'age': '年龄',
            'grade': '肿瘤分级',
            'er_positive': 'ER状态',
            'pr_positive': 'PR状态',
            'her2_positive': 'HER2状态'
        }
        
        feature_df['特征'] = feature_df['特征'].map(feature_names).fillna(feature_df['特征'])
        
        # 创建水平条形图
        fig_importance = px.bar(
            feature_df,
            x='重要性',
            y='特征',
            orientation='h',
            title="各特征对风险预测的贡献度",
            color='重要性',
            color_continuous_scale='Reds'
        )
        
        fig_importance.update_layout(
            template='plotly_white',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    def _display_recommendations(self, risk_interpretation):
        """显示建议"""
        st.subheader("💡 临床建议")
        
        recommendations = risk_interpretation.get('recommendations', [])
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
        
        # 添加免责声明
        st.markdown("""
        ---
        **⚠️ 重要提示:**
        
        - 此预测结果仅供临床参考，不能替代专业医疗诊断
        - 实际治疗方案应由专业医生根据具体情况制定
        - 预测模型基于历史数据训练，个体差异可能影响准确性
        - 建议结合其他检查结果和临床经验进行综合判断
        """)
        
        # 添加导出功能
        if st.button("📄 生成预测报告"):
            self._generate_prediction_report(risk_interpretation)
    
    def _generate_prediction_report(self, risk_interpretation):
        """生成预测报告"""
        st.success("预测报告已生成！")
        
        report_content = f"""
        # 癌症生存风险预测报告
        
        **生成时间:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## 预测结果
        - **风险评分:** {risk_interpretation.get('risk_score', 0):.1%}
        - **风险等级:** {risk_interpretation.get('risk_level', 'N/A')}
        - **风险描述:** {risk_interpretation.get('description', 'N/A')}
        
        ## 临床建议
        """
        
        recommendations = risk_interpretation.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report_content += f"\n{i}. {rec}"
        
        report_content += """
        
        ## 免责声明
        此预测结果仅供临床参考，不能替代专业医疗诊断。
        实际治疗方案应由专业医生根据具体情况制定。
        """
        
        st.download_button(
            label="下载报告",
            data=report_content,
            file_name=f"cancer_risk_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
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
        
        **模型对比**: 
        - C-index性能对比
        - Brier Score时间依赖分析
        - 集成Brier Score (IBS) 评估
        - 综合性能雷达图
        
        **生存分析**: 查看各模型的风险分层Kaplan-Meier生存曲线
        
        **风险分析**: 分析风险得分的分布和模型间相关性
        
        **交互预测**: 模拟患者特征进行实时风险预测
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### 📊 评估指标说明
        
        **C-index**: 一致性指数，衡量预测排序准确性 (0.5-1.0，越高越好)
        
        **Brier Score**: 时间依赖的预测准确性 (0-1，越低越好)
        
        **IBS**: 集成Brier Score，整体时间范围的综合性能 (0-1，越低越好)
        
        **Log-rank检验**: 风险分层统计显著性检验
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