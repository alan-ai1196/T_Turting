"""
模型训练工具模块
提供DeepSurv、Cox回归和随机生存森林模型的训练功能
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.utils import Surv
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


class DeepSurv(nn.Module):
    """DeepSurv深度学习模型"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super(DeepSurv, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_risk(self, x):
        """预测风险得分"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            risk_score = self.forward(x)
            return risk_score.cpu().numpy()


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.training_history = {}
    
    def cox_loss(self, risk_scores, durations, events):
        """Cox部分似然损失函数"""
        if risk_scores.dim() == 2:
            risk_scores = risk_scores.view(-1)
        
        event_mask = events.bool()
        if event_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        event_times = durations[event_mask]
        event_risks = risk_scores[event_mask]
        
        log_likelihood = 0.0
        
        for i, (time_i, risk_i) in enumerate(zip(event_times, event_risks)):
            at_risk_mask = durations >= time_i
            at_risk_risks = risk_scores[at_risk_mask]
            
            if len(at_risk_risks) > 0:
                log_sum_exp_risks = torch.logsumexp(at_risk_risks, dim=0)
                log_likelihood += risk_i - log_sum_exp_risks
        
        return -log_likelihood / event_mask.sum()
    
    def train_deepsurv(self, X_train, y_train_duration, y_train_event, 
                       X_test, y_test_duration, y_test_event,
                       hidden_dims=[128, 64, 32], dropout_rate=0.3,
                       batch_size=128, learning_rate=0.001, num_epochs=200, patience=20):
        """训练DeepSurv模型"""
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_duration_tensor = torch.FloatTensor(y_train_duration.values)
        y_train_event_tensor = torch.FloatTensor(y_train_event.values)
        
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_duration_tensor = torch.FloatTensor(y_test_duration.values)
        y_test_event_tensor = torch.FloatTensor(y_test_event.values)
        
        # 创建模型
        input_dim = X_train.shape[1]
        model = DeepSurv(input_dim=input_dim, 
                        hidden_dims=hidden_dims, 
                        dropout_rate=dropout_rate)
        model = model.to(self.device)
        
        # 优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # 数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_duration_tensor, y_train_event_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练历史
        train_losses = []
        train_c_indices = []
        val_c_indices = []
        best_c_index = 0
        patience_counter = 0
        
        print("开始训练DeepSurv模型...")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            
            for batch_x, batch_duration, batch_event in train_loader:
                batch_x = batch_x.to(self.device)
                batch_duration = batch_duration.to(self.device)
                batch_event = batch_event.to(self.device)
                
                optimizer.zero_grad()
                risk_scores = model(batch_x)
                loss = self.cox_loss(risk_scores, batch_duration, batch_event)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # 评估
            model.eval()
            with torch.no_grad():
                train_risks = model(X_train_tensor.to(self.device)).cpu().numpy().flatten()
                train_c_index = concordance_index(y_train_duration, -train_risks, y_train_event)
                
                val_risks = model(X_test_tensor.to(self.device)).cpu().numpy().flatten()
                val_c_index = concordance_index(y_test_duration, -val_risks, y_test_event)
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            train_c_indices.append(train_c_index)
            val_c_indices.append(val_c_index)
            
            scheduler.step(avg_loss)
            
            # 早停检查
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                      f'Train C-index: {train_c_index:.4f}, Val C-index: {val_c_index:.4f}')
            
            if patience_counter >= patience:
                print(f'早停于第 {epoch+1} 轮，最佳验证C-index: {best_c_index:.4f}')
                break
        
        # 恢复最佳模型
        model.load_state_dict(best_model_state)
        
        self.models['deepsurv'] = model
        self.training_history['deepsurv'] = {
            'train_losses': train_losses,
            'train_c_indices': train_c_indices,
            'val_c_indices': val_c_indices,
            'best_c_index': best_c_index
        }
        
        print(f"DeepSurv训练完成！最佳验证C-index: {best_c_index:.4f}")
        return model, self.training_history['deepsurv']
    
    def train_cox_regression(self, X_train, y_train_duration, y_train_event, penalizer=0.01):
        """训练Cox回归模型"""
        
        # 准备数据
        cox_train_data = X_train.copy()
        cox_train_data['Duration'] = y_train_duration
        cox_train_data['Event'] = y_train_event
        
        print("训练Cox回归模型...")
        
        # 创建和训练模型
        cox_model = CoxPHFitter(penalizer=penalizer)
        
        try:
            cox_model.fit(cox_train_data, duration_col='Duration', event_col='Event')
            print("Cox模型训练成功！")
        except Exception as e:
            print(f"Cox模型训练失败: {e}")
            print("尝试更强的正则化...")
            cox_model = CoxPHFitter(penalizer=0.1)
            cox_model.fit(cox_train_data, duration_col='Duration', event_col='Event')
            print("Cox模型训练成功（强正则化）！")
        
        self.models['cox'] = cox_model
        
        print(f"Cox模型训练完成！AIC: {cox_model.AIC_:.4f}")
        return cox_model
    
    def train_random_survival_forest(self, X_train, y_train_duration, y_train_event,
                                   n_estimators=100, max_depth=5, min_samples_split=10,
                                   min_samples_leaf=5, max_features='sqrt', random_state=42):
        """训练随机生存森林模型"""
        
        # 准备数据
        y_train_structured = Surv.from_arrays(event=y_train_event.astype(bool), 
                                            time=y_train_duration)
        
        print("训练随机生存森林模型...")
        
        # 创建和训练模型
        rsf_model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
        rsf_model.fit(X_train, y_train_structured)
        
        self.models['rsf'] = rsf_model
        
        print("随机生存森林模型训练完成！")
        return rsf_model
    
    def evaluate_models(self, X_test, y_test_duration, y_test_event):
        """评估所有模型"""
        results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'deepsurv':
                # DeepSurv评估
                X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
                with torch.no_grad():
                    risk_scores = model(X_test_tensor).cpu().numpy().flatten()
                c_index = concordance_index(y_test_duration, -risk_scores, y_test_event)
                
            elif model_name == 'cox':
                # Cox回归评估
                cox_test_data = X_test.copy()
                cox_test_data['Duration'] = y_test_duration
                cox_test_data['Event'] = y_test_event
                risk_scores = model.predict_partial_hazard(cox_test_data)
                c_index = concordance_index(y_test_duration, risk_scores, y_test_event)
                
            elif model_name == 'rsf':
                # RSF评估
                risk_scores = model.predict(X_test)
                c_index = concordance_index_censored(
                    y_test_event.astype(bool), y_test_duration, risk_scores
                )[0]
            
            results[model_name] = {
                'c_index': c_index,
                'risk_scores': risk_scores
            }
        
        return results
    
    def save_models(self, save_dir):
        """保存所有模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'deepsurv':
                torch.save(model.state_dict(), save_dir / 'deepsurv_model.pth')
            else:
                with open(save_dir / f'{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # 保存训练历史
        with open(save_dir / 'training_history.pkl', 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"所有模型已保存至: {save_dir}")
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if 'deepsurv' not in self.training_history:
            print("没有DeepSurv训练历史可绘制")
            return
        
        history = self.training_history['deepsurv']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(history['train_losses'], label='Training Loss', color='blue')
        axes[0].set_title('训练损失曲线')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cox Partial Likelihood Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # C-index曲线
        axes[1].plot(history['train_c_indices'], label='Training C-index', color='blue')
        axes[1].plot(history['val_c_indices'], label='Validation C-index', color='red')
        axes[1].set_title('C-index变化曲线')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('C-index')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """主函数示例"""
    # 加载数据
    data_dir = Path('../data/processed')
    train_data = pd.read_csv(data_dir / 'train_data.csv')
    test_data = pd.read_csv(data_dir / 'test_data.csv')
    
    # 加载预处理器
    with open(data_dir / 'preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    
    feature_columns = preprocessors['feature_columns']
    
    # 准备数据
    X_train = train_data[feature_columns]
    y_train_duration = train_data['Duration']
    y_train_event = train_data['Event']
    
    X_test = test_data[feature_columns]
    y_test_duration = test_data['Duration']
    y_test_event = test_data['Event']
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(device=device)
    
    # 训练所有模型
    print("开始训练所有模型...")
    
    # 训练DeepSurv
    deepsurv_model, history = trainer.train_deepsurv(
        X_train, y_train_duration, y_train_event,
        X_test, y_test_duration, y_test_event
    )
    
    # 训练Cox回归
    cox_model = trainer.train_cox_regression(X_train, y_train_duration, y_train_event)
    
    # 训练随机生存森林
    rsf_model = trainer.train_random_survival_forest(X_train, y_train_duration, y_train_event)
    
    # 评估模型
    results = trainer.evaluate_models(X_test, y_test_duration, y_test_event)
    
    print("\\n模型评估结果:")
    for model_name, result in results.items():
        print(f"{model_name}: C-index = {result['c_index']:.4f}")
    
    # 保存模型
    trainer.save_models('../model')
    
    # 绘制训练曲线
    trainer.plot_training_curves('../reports/training_curves.png')
    
    print("所有模型训练完成！")


if __name__ == "__main__":
    main()