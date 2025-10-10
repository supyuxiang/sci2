import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
try:
    import CoolProp.CoolProp as CP
    print("CoolProp 加载成功，将使用实时属性查询。")
except ImportError:
    print("CoolProp 未安装，将使用拟合属性函数作为 fallback。")
    CP = None

# 文件路径
csv_file = r"C:\Users\31346\Desktop\pinnnn.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"文件 '{csv_file}' 不存在。")

# 加载数据
try:
    df = pd.read_csv(csv_file, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, encoding='gbk')
df = df.dropna()

# 提取数据
try:
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    T_data = df['T (K)'].values
    p_data = df['p (Pa)'].values
    w_data = df['w (m/s)'].values
except KeyError as e:
    raise KeyError(f"CSV 文件中缺少列：{str(e)}。")

# 数据归一化
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
z_min, z_max = np.min(z), np.max(z)
T_min, T_max = np.min(T_data), np.max(T_data)
p_min, p_max = np.min(p_data), np.max(p_data)
w_min, w_max = np.min(w_data), np.max(w_data)

x_norm = (x - x_min) / (x_max - x_min + 1e-8)
y_norm = (y - y_min) / (y_max - y_min + 1e-8)
z_norm = (z - z_min) / (z_max - z_min + 1e-8)
T_norm = (T_data - T_min) / (T_max - T_min + 1e-8)
p_norm = (p_data - p_min) / (p_max - p_min + 1e-8)
w_norm = (w_data - w_min) / (w_max - w_min + 1e-8)

inputs = torch.tensor(np.stack([x_norm, y_norm, z_norm], axis=1), dtype=torch.float32, requires_grad=True)
T_data = torch.tensor(T_norm, dtype=torch.float32)
p_data = torch.tensor(p_norm, dtype=torch.float32)
w_data = torch.tensor(w_norm, dtype=torch.float32)

print(f"输入张量形状: {inputs.shape}")
print(f"归一化后温度范围: [{T_min:.2f}, {T_max:.2f}] K")
print(f"归一化后压力范围: [{p_min:.2f}, {p_max:.2f}] Pa")

# 定义 PINN 模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

# 材料属性函数
def rho_func(T, P=3e6):
    if CP is not None:
        return torch.tensor([CP.PropsSI('D', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, P)],
                           dtype=torch.float32, device=T.device)
    else:
        return -0.005343 * T**2 + 3.0787 * T + 283.56

def mu_func(T, P=3e6):
    if CP is not None:
        return torch.tensor([CP.PropsSI('V', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, P)],
                           dtype=torch.float32, device=T.device)
    else:
        return torch.exp(-0.0091114 * T - 4.3961)

def Cp_func(T, P=3e6):
    if CP is not None:
        return torch.tensor([CP.PropsSI('C', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, P)],
                           dtype=torch.float32, device=T.device)
    else:
        return 4.1587 * T + 947.66

def k_func(T, P=3e6):
    if CP is not None:
        return torch.tensor([CP.PropsSI('L', 'T', t.item(), 'P', p.item(), 'n-Decane') for t, p in zip(T, P)],
                           dtype=torch.float32, device=T.device)
    else:
        return torch.tensor(0.133, device=T.device)

# 训练循环
num_epochs = 15000
lambda_pde = 1e-2
lambda_boundary = 1e-3

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)
    T_pred = outputs[:, 0]
    p_pred = outputs[:, 1]
    w_pred = outputs[:, 2]

    # 数据损失
    loss_data = mse(T_pred, T_data) + mse(p_pred, p_data) + mse(w_pred, w_data)

    # 计算导数
    ones = torch.ones_like(w_pred)
    T_x, T_y, T_z = torch.autograd.grad(T_pred, inputs, grad_outputs=ones, create_graph=True)[0].T
    p_x, p_y, p_z = torch.autograd.grad(p_pred, inputs, grad_outputs=ones, create_graph=True)[0].T
    w_x, w_y, w_z = torch.autograd.grad(w_pred, inputs, grad_outputs=ones, create_graph=True)[0].T

    T_xx = torch.autograd.grad(T_x, inputs, ones, create_graph=True)[0][:, 0]
    T_yy = torch.autograd.grad(T_y, inputs, ones, create_graph=True)[0][:, 1]
    T_zz = torch.autograd.grad(T_z, inputs, ones, create_graph=True)[0][:, 2]
    w_xx = torch.autograd.grad(w_x, inputs, ones, create_graph=True)[0][:, 0]
    w_yy = torch.autograd.grad(w_y, inputs, ones, create_graph=True)[0][:, 1]
    w_zz = torch.autograd.grad(w_z, inputs, ones, create_graph=True)[0][:, 2]
    w_xz = torch.autograd.grad(w_x, inputs, ones, create_graph=True)[0][:, 2]
    w_yz = torch.autograd.grad(w_y, inputs, ones, create_graph=True)[0][:, 2]

    # 反归一化
    T_phys = T_pred * (T_max - T_min) + T_min
    p_phys = p_pred * (p_max - p_min) + p_min

    # 实时属性查询
    rho_pred = rho_func(T_phys, p_phys)
    mu_pred = mu_func(T_phys, p_phys)
    Cp_pred = Cp_func(T_phys, p_phys)
    k_pred = k_func(T_phys, p_phys)

    # 连续性方程
    rho_w_z = torch.autograd.grad(rho_pred * w_pred, inputs, ones, create_graph=True)[0][:, 2]
    loss_cont = (rho_w_z ** 2).mean()

    # z 方向动量方程
    mu_w_x_x = torch.autograd.grad(mu_pred * w_x, inputs, ones, create_graph=True)[0][:, 0]
    mu_w_y_y = torch.autograd.grad(mu_pred * w_y, inputs, ones, create_graph=True)[0][:, 1]
    mu_w_z_z = torch.autograd.grad(mu_pred * w_z, inputs, ones, create_graph=True)[0][:, 2]
    loss_mom_z = ((rho_pred * w_pred * w_z + p_z - mu_w_x_x - mu_w_y_y - mu_w_z_z) ** 2).mean()

    # x 和 y 方向动量
    loss_mom_x = ((p_x - mu_pred * w_xz) ** 2).mean()
    loss_mom_y = ((p_y - mu_pred * w_yz) ** 2).mean()

    # 能量方程
    loss_energy = ((rho_pred * Cp_pred * w_pred * T_z - k_pred * (T_xx + T_yy + T_zz)) ** 2).mean()

    # 总物理损失
    loss_pde = loss_cont + loss_mom_z + loss_mom_x + loss_mom_y + loss_energy

    # 出口压力边界 (p=3 MPa)
    p_out_norm = (3e6 - p_min) / (p_max - p_min + 1e-8)
    boundary_mask = (inputs[:, 2] >= 0.95)
    loss_boundary = torch.tensor(0.0, requires_grad=True)
    if boundary_mask.sum() > 0:
        p_boundary = p_pred[boundary_mask]
        loss_boundary = mse(p_boundary, torch.full_like(p_boundary, p_out_norm))

    # 入口温度 (T=300 K)
    T_in_norm = (300 - T_min) / (T_max - T_min + 1e-8)
    inlet_mask = (inputs[:, 2] <= 0.001)
    loss_inlet_T = torch.tensor(0.0, requires_grad=True)
    if inlet_mask.sum() > 0:
        T_inlet = T_pred[inlet_mask]
        loss_inlet_T = mse(T_inlet, torch.full_like(T_inlet, T_in_norm))

    # 总损失 (移除热流和质量流量约束)
    loss = loss_data + lambda_pde * loss_pde + lambda_boundary * (loss_boundary + loss_inlet_T)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 打印
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: 总损失 = {loss.item():.4e}, 数据损失 = {loss_data.item():.4e}, "
              f"物理损失 = {loss_pde.item():.4e}, 出口压力损失 = {loss_boundary.item():.4e}, "
              f"入口温度损失 = {loss_inlet_T.item():.4e}")

# 训练后预测与评价
model.eval()
with torch.no_grad():
    # 训练点预测
    outputs = model(inputs)
    T_pred = outputs[:, 0] * (T_max - T_min) + T_min
    p_pred = outputs[:, 1] * (p_max - p_min) + p_min
    w_pred = outputs[:, 2] * (w_max - w_min) + w_min

    # 网格点预测
    n_grid = 5
    x_grid = torch.linspace(0, 1, n_grid)
    y_grid = torch.linspace(0, 1, n_grid)
    z_grid = torch.linspace(0, 1, n_grid)
    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    grid_outputs = model(grid_points)
    T_grid = grid_outputs[:, 0] * (T_max - T_min) + T_min
    p_grid = grid_outputs[:, 1] * (p_max - p_min) + p_min
    w_grid = grid_outputs[:, 2] * (w_max - w_min) + w_min

# 计算 MAE 和 RMSE
mae_T = np.mean(np.abs(T_pred.numpy() - (T_data.numpy() * (T_max - T_min) + T_min)))
rmse_T = np.sqrt(np.mean((T_pred.numpy() - (T_data.numpy() * (T_max - T_min) + T_min))**2))
mae_p = np.mean(np.abs(p_pred.numpy() - (p_data.numpy() * (p_max - p_min) + p_min)))
rmse_p = np.sqrt(np.mean((p_pred.numpy() - (p_data.numpy() * (p_max - p_min) + p_min))**2))
mae_w = np.mean(np.abs(w_pred.numpy() - (w_data.numpy() * (w_max - w_min) + w_min)))
rmse_w = np.sqrt(np.mean((w_pred.numpy() - (w_data.numpy() * (w_max - w_min) + w_min))**2))

print(f"温度 MAE: {mae_T:.4f} K, RMSE: {rmse_T:.4f} K")
print(f"压力 MAE: {mae_p:.4f} Pa, RMSE: {rmse_p:.4f} Pa")
print(f"速度 MAE: {mae_w:.4f} m/s, RMSE: {rmse_w:.4f} m/s")

# 可视化
plt.figure(figsize=(15, 10))

# 训练点预测 vs 实际
plt.subplot(2, 3, 1)
plt.scatter(T_data.numpy() * (T_max - T_min) + T_min, T_pred.numpy(), alpha=0.5)
plt.plot([T_min, T_max], [T_min, T_max], 'r--')
plt.xlabel('实际温度 (K)')
plt.ylabel('预测温度 (K)')
plt.title('训练点：温度预测 vs 实际')

plt.subplot(2, 3, 2)
plt.scatter(p_data.numpy() * (p_max - p_min) + p_min, p_pred.numpy(), alpha=0.5)
plt.plot([p_min, p_max], [p_min, p_max], 'r--')
plt.xlabel('实际压力 (Pa)')
plt.ylabel('预测压力 (Pa)')
plt.title('训练点：压力预测 vs 实际')

plt.subplot(2, 3, 3)
plt.scatter(w_data.numpy() * (w_max - w_min) + w_min, w_pred.numpy(), alpha=0.5)
plt.plot([w_min, w_max], [w_min, w_max], 'r--')
plt.xlabel('实际速度 (m/s)')
plt.ylabel('预测速度 (m/s)')
plt.title('训练点：速度预测 vs 实际')

# 网格点预测
plt.subplot(2, 3, 4)
plt.scatter(grid_points[:, 2].numpy() * (z_max - z_min) + z_min, T_grid.numpy(), alpha=0.5)
plt.xlabel('z (m)')
plt.ylabel('预测温度 (K)')
plt.title('网格点：温度分布')

plt.subplot(2, 3, 5)
plt.scatter(grid_points[:, 2].numpy() * (z_max - z_min) + z_min, p_grid.numpy(), alpha=0.5)
plt.xlabel('z (m)')
plt.ylabel('预测压力 (Pa)')
plt.title('网格点：压力分布')

plt.subplot(2, 3, 6)
plt.scatter(grid_points[:, 2].numpy() * (z_max - z_min) + z_min, w_grid.numpy(), alpha=0.5)
plt.xlabel('z (m)')
plt.ylabel('预测速度 (m/s)')
plt.title('网格点：速度分布')

plt.tight_layout()
plt.savefig('prediction_vs_actual_and_grid.png')
plt.show()