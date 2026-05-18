import matplotlib
matplotlib.use("Agg")
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# ---------- 固定随机种子，保证结果可复现 ----------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
dde.config.set_random_seed(seed)

# ==========================================
# 第一部分：定义偏微分方程 (PDE)
# ==========================================
def pde(x, y):
    # x 是输入张量，包含空间 x 和 时间 t。即 x[:, 0] 是空间，x[:, 1] 是时间。
    # y 是神经网络的输出，即预测的温度 u。
    dy_t = dde.grad.jacobian(y, x, i=0, j=1) 
    dy_xx = dde.grad.hessian(y, x, i=0, j=0) 
    # 当神经网络完美符合物理规律时，这个返回值应该趋近于 0
    alpha = 0.4
    return dy_t - alpha * dy_xx

# ==========================================
# 第二部分：定义几何空间与时间域
# ==========================================
# 空间域：一根从 x=-1 到 x=1 的一维金属棒
geom = dde.geometry.Interval(-1, 1)

# 时间域：从 t=0 到 t=1 秒
timedomain = dde.geometry.TimeDomain(0, 1)

# 时空联合域：把空间和时间组合起来形成完整的研究区域
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ==========================================
# 第三部分：定义边界条件(BC)和初始条件(IC)
# ==========================================
# 边界条件 (DirichletBC)：金属棒的两端 (x=-1 和 x=1) 温度始终保持为 0
# lambda x: 0 表示目标值为 0
bc = dde.icbc.DirichletBC(
    geomtime, 
    lambda x: 0, 
    lambda _, on_boundary: on_boundary
)

# 初始条件 (IC)：在 t=0 时刻，金属棒上的温度分布符合正弦曲线 sin(pi * x)
# x[:, 0:1] 是提取所有的空间坐标点进行 numpy 计算
ic = dde.icbc.IC(
    geomtime, 
    lambda x: np.sin(np.pi * x[:, 0:1]), 
    lambda _, on_initial: on_initial
)

# ==========================================
# 第四部分：生成训练数据对象
# ==========================================
data = dde.data.TimePDE(
    geomtime,      
    pde,           
    [bc, ic],      
    num_domain=2000,    # 在时空域内部随机采样 2000 个点（用来计算 PDE 残差）
    num_boundary=80,   # 在边界上采样 80 个点（用来计算 BC 误差）
    num_initial=80,    # 在 t=0 时刻采样 80 个点（用来计算 IC 误差）
)

# ==========================================
# 第五部分：定义神经网络架构
# ==========================================
# [2, 50, 50, 50, 50, 1] 
# 输入层 2 个神经元 (x, t) -> 4 个隐藏层，每层 50 个神经元 -> 输出层 1 个神经元 (u)
layer_size = [2] + [50] * 4 + [1]
activation = "tanh"        # 激活函数使用 tanh，因为求导平滑
initializer = "Glorot normal" # 经典的权重初始化方法

net = dde.nn.FNN(layer_size, activation, initializer)

# ==========================================
# 第六部分：编译与训练模型
# ==========================================
# 操作 : 去掉 PDE 残差，退化为纯 NN
# ==========================================
model = dde.Model(data, net)

# 核心魔法在这里：loss_weights 对应 [PDE残差, 边界条件BC, 初始条件IC]
# 把第一个值设为 0，意味着模型训练时完全不看物理公式！
model.compile("adam", lr=1e-3, loss_weights=[0, 1, 1])

print("开始训练纯 NN 模型 (PDE 物理约束已关闭，仅保留 BC 和 IC)...")
losshistory, train_state = model.train(iterations=5000)

# ==========================================
# 第七部分：结果可视化与保存
# ==========================================
steps = losshistory.steps
train_loss = [sum(loss) for loss in losshistory.loss_train]

# --- 损失曲线图 ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(steps, train_loss, "b.-", markersize=4, linewidth=1.2, label="Train Loss")
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_title("Pure NN Training Loss (1D Heat Equation)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/NN_loss.png", dpi=150)
plt.close()

# --- 预测热力图 ---
dde.utils.plot_best_state(train_state)
plt.savefig("results/NN_prediction.png")
plt.close()

# --- 保存损失历史到文件 ---
with open("results/NN_loss_history.txt", "w") as f:
    f.write("step,total_loss\n")
    for s, loss in zip(steps, train_loss):
        f.write(f"{s},{loss}\n")

print("训练完成！所有结果已保存至 results/ 目录：")
print("  - results/NN_loss.png          (损失曲线)")
print("  - results/NN_prediction.png    (预测热力图)")
print("  - results/NN_loss_history.txt  (损失历史)")
