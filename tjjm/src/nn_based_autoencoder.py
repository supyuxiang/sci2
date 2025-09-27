# autoencoder_train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# 导入数据预处理函数
from data_preprocessing import data_preprocessing_0
# 导入SwanLab配置
from swanlab_config import init_swanlab, log_metrics, log_model_params, finish_run

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("husl", 8)




# 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder():
    # 初始化SwanLab
    run = init_swanlab("nn_based_autoencoder.py")
    
    print("="*60)
    print("Autoencoder Training Analysis")
    print("="*60)
    
    # 创建结果目录
    results_dir = "../results/autoencoder/autoencoder"
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取预处理后的数据
    x_scaled, y_scaled, T_array = data_preprocessing_0()
    
    # 数据准备
    X = np.column_stack([x_scaled, y_scaled])
    print(f"Feature matrix X shape: {X.shape}")
    
    # 数据分割
    X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42, shuffle=True)
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转换为PyTorch张量
    train_data = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(X_train_scaled))
    test_data = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(X_test_scaled))
    
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
            
    # 模型训练
    input_dim = X_train.shape[1]
    encoding_dim = 2  # 编码维度

    # 设备选择：优先使用 cuda:8
    if torch.cuda.is_available():
        device = torch.device("cuda:8")
        try:
            torch.cuda.set_device(8)
        except Exception:
            pass
    else:
        device = torch.device("cpu")

    model = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Autoencoder model training...")
    num_epochs = 150
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # 评估模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            
    test_loss = test_loss / len(test_loader.dataset)
    print(f'\nTest Loss: {test_loss:.4f}')
    
    # 记录模型参数和指标到SwanLab
    model_params = {
        'model_type': 'autoencoder',
        'input_dim': input_dim,
        'encoding_dim': encoding_dim,
        'epochs': num_epochs,
        'learning_rate': 0.001,
        'batch_size': 256,
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0]
    }
    log_model_params(run, model_params)
    
    metrics = {
        'Test_Loss': test_loss,
        'Training_Loss': train_losses[-1]
    }
    log_metrics(run, metrics)
    
    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.title('Autoencoder Training Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 提取编码特征
    with torch.no_grad():
        enc_in = torch.FloatTensor(scaler.transform(X)).to(device)
        encoded_features = model.encoder(enc_in).cpu().numpy()
    
    # 保存结果
    np.save(os.path.join(results_dir, 'encoded_features.npy'), encoded_features)
    print(f"\nEncoded features saved to: {results_dir}")
    print("Autoencoder training completed!")
    
    # 完成SwanLab运行
    finish_run(run)
    
    # 返回训练好的模型和scaler，供Final_model使用
    return model, scaler

class Final_model(nn.Module):
    def __init__(self, pre_trained_encoder, out_dim, encoding_dim):
        super(Final_model, self).__init__()
        # 使用预训练的encoder
        self.encoder = pre_trained_encoder
        # 冻结encoder参数，只训练后续网络
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.network = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 4),
            nn.BatchNorm1d(encoding_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(encoding_dim * 4, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.BatchNorm1d(encoding_dim // 2),
            nn.Tanh(),
            
            nn.Linear(encoding_dim // 2, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return self.network(x).squeeze(-1)

def train_final_model_based_autoencoder(pre_trained_autoencoder=None, pre_trained_scaler=None):
    # 初始化SwanLab
    run = init_swanlab("nn_based_autoencoder.py")
    
    print("="*60)
    print("Final Model_based_autoencoder Training Analysis")
    print("="*60)

    # 创建结果目录
    results_dir = "../results/autoencoder/final_model_based_autoencoder"
    os.makedirs(results_dir, exist_ok=True)

    # 获取预处理后的数据
    x_scaled, y_scaled, T_array = data_preprocessing_0()

    # 数据准备
    X = np.column_stack([x_scaled, y_scaled])
    print(f"Feature matrix X shape: {X.shape}")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, T_array, test_size=0.2, random_state=42, shuffle=True)

    # 使用预训练的scaler或创建新的scaler
    if pre_trained_scaler is not None:
        scaler = pre_trained_scaler
        print("Using pre-trained scaler for feature normalization")
    else:
        scaler = StandardScaler()
        print("Creating new scaler for feature normalization")
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为PyTorch张量
    train_data = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # 定义模型
    input_dim = X_train.shape[1]
    out_dim = 1
    encoding_dim = 2
    
    # 设备选择：优先使用 cuda:8
    if torch.cuda.is_available():
        device = torch.device("cuda:8")
        try:
            torch.cuda.set_device(8)
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    
    # 使用预训练的encoder或创建新的
    if pre_trained_autoencoder is not None:
        print("Using pre-trained encoder from autoencoder training")
        model = Final_model(pre_trained_autoencoder.encoder, out_dim, encoding_dim).to(device)
    else:
        print("Creating new encoder (not recommended)")
        # 创建一个临时的autoencoder来获取encoder
        temp_autoencoder = Autoencoder(input_dim, encoding_dim)
        model = Final_model(temp_autoencoder.encoder, out_dim, encoding_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting Final Model_based_autoencoder model training...")
    num_epochs = 500
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            
    # 评估模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            
    test_loss = test_loss / len(test_loader.dataset)
    print(f'\nTest Loss: {test_loss:.4f}')
    
    # 记录模型参数和指标到SwanLab
    model_params = {
        'model_type': 'autoencoder_based_final_model',
        'input_dim': input_dim,
        'encoding_dim': encoding_dim,
        'out_dim': out_dim,
        'epochs': num_epochs,
        'learning_rate': 0.001,
        'batch_size': 256,
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0],
        'using_pretrained_encoder': pre_trained_autoencoder is not None
    }
    log_model_params(run, model_params)
    
    metrics = {
        'Test_Loss': test_loss,
        'Final_Training_Loss': train_losses[-1]
    }
    log_metrics(run, metrics)

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.title('Final Model_based_autoencoder Training Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'training_loss_based_autoencoder.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # 提取编码特征
    with torch.no_grad():
        enc_in = torch.FloatTensor(scaler.transform(X)).to(device)
        encoded_features = model.encoder(enc_in).cpu().numpy()

    # 保存结果
    np.save(os.path.join(results_dir, 'encoded_features_based_autoencoder.npy'), encoded_features)
    print(f"\nEncoded features saved to: {results_dir}")
    print("Final Model_based_autoencoder training completed!")
    
    # 完成SwanLab运行
    finish_run(run)

if __name__ == "__main__":
    # 首先训练autoencoder并获取训练好的模型
    print("Step 1: Training Autoencoder...")
    trained_autoencoder, trained_scaler = train_autoencoder()
    
    # 然后使用训练好的encoder训练Final_model
    print("\nStep 2: Training Final Model using pre-trained encoder...")
    train_final_model_based_autoencoder(pre_trained_autoencoder=trained_autoencoder, 
                                       pre_trained_scaler=trained_scaler)