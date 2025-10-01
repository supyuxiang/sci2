import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 获取当前路径
current_path = Path(__file__).resolve()
print(current_path)
print(current_path.parent.parent)

# 构建数据路径
data_path = current_path.parent.parent / 'data' / '三维双热通量.xlsx'

# 读取Excel文件
data = pd.read_excel(data_path)

# 定义索引范围
idx_start = 8
idx_end = 15698

# 提取数据并确保为数值类型
x = pd.to_numeric(data.iloc[idx_start:idx_end, 0], errors='coerce')
y = pd.to_numeric(data.iloc[idx_start:idx_end, 6], errors='coerce')

# 创建保存目录
save_dir = current_path.parent.parent / 'outputs' / 'raw_data_visulize'
save_dir.mkdir(parents=True, exist_ok=True)

# 绘制XY散点图
plt.figure(figsize=(10, 12))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Raw Data')
plt.savefig(save_dir / 'raw_data_xy_visulize.png', dpi=300)
plt.close()  # 关闭图形释放内存

# 提取其他数据列
u = pd.to_numeric(data.iloc[idx_start:idx_end, 4], errors='coerce')
v = pd.to_numeric(data.iloc[idx_start:idx_end, 5], errors='coerce')
w = pd.to_numeric(data.iloc[idx_start:idx_end, 6], errors='coerce')
p = pd.to_numeric(data.iloc[idx_start:idx_end, 7], errors='coerce')
T = pd.to_numeric(data.iloc[idx_start:, 3], errors='coerce')  # 修复索引范围一致

def visulize_one(data, name: str):
    """可视化单个数据列"""
    try:
        plt.figure(figsize=(10, 12))
        plt.plot(data.values)  # 使用.values确保是数值数组
        plt.title(f'Raw Data: {name}')
        plt.xlabel('index')
        plt.ylabel(name)
        plt.savefig(save_dir / f'{name}.png', dpi=300)
        plt.close()  # 关闭图形释放内存
        print(f"Successfully visualized {name}")
    except Exception as e:
        print(f"Error visualizing {name}: {str(e)}")

# 可视化各列数据
visulize_one(u, 'u')
visulize_one(v, 'v')
visulize_one(w, 'w')
visulize_one(p, 'p')
visulize_one(T, 'T')

print("All visualizations completed!")