"""
数据预处理工具模块
提供癌症患者数据集的数据清洗、特征工程和预处理功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser as parser
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class CancerDataPreprocessor:
    """癌症患者数据预处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.numerical_cols = []
        self.categorical_cols = []
    
    def load_data(self, data_path):
        """加载数据"""
        self.df = pd.read_csv(data_path)
        return self.df
    
    def clean_missing_values(self, df):
        """清理非标准缺失值"""
        df_clean = df.copy()
        non_standard_missing = ['N/A', '', 'None', 'null', 'NULL', ' ']
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].replace(non_standard_missing, np.nan)
        
        return df_clean
    
    def parse_date_safe(self, date_str):
        """安全解析日期"""
        if pd.isna(date_str):
            return None
        try:
            return parser.parse(str(date_str))
        except:
            return None
    
    def create_survival_variables(self, df):
        """创建生存分析变量"""
        df_processed = df.copy()
        
        # 转换日期字段
        df_processed['DiagnosisDate'] = df_processed['DiagnosisDate'].apply(self.parse_date_safe)
        df_processed['SurgeryDate'] = df_processed['SurgeryDate'].apply(self.parse_date_safe)
        
        # 生存状态转换为事件指示器
        df_processed['Event'] = (df_processed['SurvivalStatus'] == 'Deceased').astype(int)
        
        # 生存时间
        df_processed['Duration'] = df_processed['FollowUpMonths']
        
        return df_processed
    
    def feature_engineering(self, df):
        """特征工程"""
        df_engineered = df.copy()
        
        # 年龄分组
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                                         bins=[0, 30, 50, 70, 100], 
                                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # 肿瘤大小分组
        df_engineered['TumorSizeGroup'] = pd.cut(df_engineered['TumorSize'],
                                               bins=[0, 5, 10, 15, float('inf')],
                                               labels=['Small', 'Medium', 'Large', 'VeryLarge'])
        
        # 治疗强度指标
        df_engineered['TreatmentIntensity'] = (df_engineered['ChemotherapySessions'] + 
                                             df_engineered['RadiationSessions'])
        
        # 二元特征
        df_engineered['HasSurgery'] = (~df_engineered['SurgeryDate'].isna()).astype(int)
        df_engineered['HasComorbidities'] = (~df_engineered['Comorbidities'].isna()).astype(int)
        df_engineered['HasGeneticMutation'] = (~df_engineered['GeneticMutation'].isna()).astype(int)
        
        return df_engineered
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        df_filled = df.copy()
        
        # 分类变量用众数填充
        categorical_features = ['Gender', 'Province', 'Ethnicity', 'TumorType', 'CancerStage', 
                               'Metastasis', 'TreatmentType', 'SmokingStatus', 'AlcoholUse']
        
        for col in categorical_features:
            if col in df_filled.columns:
                mode_value = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else 'Unknown'
                df_filled[col] = df_filled[col].fillna(mode_value)
        
        # 数值变量用中位数填充
        numerical_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions']
        
        for col in numerical_features:
            if col in df_filled.columns:
                median_value = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_value)
        
        return df_filled
    
    def encode_features(self, df):
        """编码特征"""
        df_encoded = df.copy()
        
        # 有序分类变量
        ordinal_features = {
            'CancerStage': ['I', 'II', 'III', 'IV'],
            'AgeGroup': ['Young', 'Middle', 'Senior', 'Elderly'],
            'TumorSizeGroup': ['Small', 'Medium', 'Large', 'VeryLarge']
        }
        
        for feature, order in ordinal_features.items():
            if feature in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[feature] = df_encoded[feature].astype('category')
                df_encoded[feature] = df_encoded[feature].cat.reorder_categories(order, ordered=True)
                df_encoded[feature + '_encoded'] = le.fit_transform(df_encoded[feature])
                self.label_encoders[feature] = le
        
        # 无序分类变量 - One-Hot编码
        nominal_features = ['Gender', 'Province', 'Ethnicity', 'TumorType', 'Metastasis', 
                           'TreatmentType', 'SmokingStatus', 'AlcoholUse']
        
        for feature in nominal_features:
            if feature in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        return df_encoded
    
    def prepare_modeling_data(self, df_encoded):
        """准备建模数据"""
        # 选择特征
        feature_columns = [
            'Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions',
            'TreatmentIntensity', 'HasSurgery', 'HasComorbidities', 'HasGeneticMutation',
            'CancerStage_encoded', 'AgeGroup_encoded', 'TumorSizeGroup_encoded'
        ]
        
        # 添加哑变量
        dummy_columns = [col for col in df_encoded.columns if any(prefix in col for prefix in 
                        ['Gender_', 'Province_', 'Ethnicity_', 'TumorType_', 'Metastasis_', 
                         'TreatmentType_', 'SmokingStatus_', 'AlcoholUse_'])]
        
        feature_columns.extend(dummy_columns)
        
        # 检查可用特征
        available_features = [col for col in feature_columns if col in df_encoded.columns]
        self.feature_columns = available_features
        
        # 目标变量
        target_columns = ['Duration', 'Event']
        
        # 创建建模数据集
        modeling_data = df_encoded[available_features + target_columns].copy()
        modeling_data = modeling_data.dropna()
        
        return modeling_data
    
    def scale_features(self, X):
        """特征缩放"""
        self.numerical_cols = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'TreatmentIntensity']
        self.categorical_cols = [col for col in self.feature_columns if col not in self.numerical_cols]
        
        X_scaled = X.copy()
        X_scaled[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        
        return X_scaled
    
    def split_data(self, X, y_duration, y_event, test_size=0.2, random_state=42):
        """数据分割"""
        return train_test_split(X, y_duration, y_event, 
                               test_size=test_size, 
                               random_state=random_state, 
                               stratify=y_event)
    
    def save_preprocessor(self, save_path):
        """保存预处理器"""
        preprocessor_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_dict, f)
    
    def load_preprocessor(self, load_path):
        """加载预处理器"""
        with open(load_path, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        
        self.scaler = preprocessor_dict['scaler']
        self.label_encoders = preprocessor_dict['label_encoders']
        self.feature_columns = preprocessor_dict['feature_columns']
        self.numerical_cols = preprocessor_dict['numerical_cols']
        self.categorical_cols = preprocessor_dict['categorical_cols']
    
    def process_pipeline(self, data_path, save_dir):
        """完整的预处理流水线"""
        # 加载数据
        df = self.load_data(data_path)
        print(f"原始数据形状: {df.shape}")
        
        # 清理缺失值
        df_clean = self.clean_missing_values(df)
        
        # 创建生存变量
        df_survival = self.create_survival_variables(df_clean)
        
        # 特征工程
        df_engineered = self.feature_engineering(df_survival)
        
        # 处理缺失值
        df_filled = self.handle_missing_values(df_engineered)
        
        # 编码特征
        df_encoded = self.encode_features(df_filled)
        
        # 准备建模数据
        modeling_data = self.prepare_modeling_data(df_encoded)
        print(f"建模数据形状: {modeling_data.shape}")
        
        # 准备特征和目标变量
        X = modeling_data[self.feature_columns]
        y_duration = modeling_data['Duration']
        y_event = modeling_data['Event']
        
        # 特征缩放
        X_scaled = self.scale_features(X)
        
        # 数据分割
        X_train, X_test, y_train_duration, y_test_duration, y_train_event, y_test_event = self.split_data(
            X_scaled, y_duration, y_event)
        
        # 保存数据
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 保存处理后的数据
        df_filled.to_csv(save_dir / 'cleaned_data.csv', index=False)
        df_encoded.to_csv(save_dir / 'encoded_data.csv', index=False)
        modeling_data.to_csv(save_dir / 'modeling_data.csv', index=False)
        
        # 保存训练测试集
        train_data = pd.concat([
            X_train, 
            pd.DataFrame({'Duration': y_train_duration, 'Event': y_train_event}, index=X_train.index)
        ], axis=1)
        
        test_data = pd.concat([
            X_test, 
            pd.DataFrame({'Duration': y_test_duration, 'Event': y_test_event}, index=X_test.index)
        ], axis=1)
        
        train_data.to_csv(save_dir / 'train_data.csv', index=False)
        test_data.to_csv(save_dir / 'test_data.csv', index=False)
        
        # 保存预处理器
        self.save_preprocessor(save_dir / 'preprocessors.pkl')
        
        print(f"数据预处理完成！文件保存至: {save_dir}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'feature_columns': self.feature_columns,
            'modeling_data': modeling_data
        }


def main():
    """主函数示例"""
    preprocessor = CancerDataPreprocessor()
    
    # 设置路径
    data_path = '../data/raw/dataset.csv'
    save_dir = '../data/processed'
    
    # 运行预处理流水线
    results = preprocessor.process_pipeline(data_path, save_dir)
    
    print("预处理完成！")
    print(f"训练集大小: {results['train_data'].shape}")
    print(f"测试集大小: {results['test_data'].shape}")
    print(f"特征数量: {len(results['feature_columns'])}")


if __name__ == "__main__":
    main()