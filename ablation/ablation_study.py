# 设置标准输出编码为 UTF-8，避免 Windows 下 emoji 等字符的编码报错
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入 DeepXDE 深度学习求解微分方程的核心库
import deepxde as dde
# 导入 NumPy 用于数值计算和数组操作
import numpy as np
# 导入 time 用于计时训练耗时
import time
# 导入 matplotlib 用于绘制实验结果图表
import matplotlib.pyplot as plt

# 解决 matplotlib 中文显示问题：使用 Windows 自带的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 提高 matplotlib 画图的清晰度
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 1. 定义物理方程：一维热传导方程 ∂u/∂t = 0.4 * ∂²u/∂x²
#    (alpha = 0.4 为热扩散系数)
# ==========================================
def pde(x, y):
    # dy_t: y 对 t（x 的第1列，索引 j=1）的一阶偏导
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    # dy_xx: y 对 x（x 的第0列，索引 j=0）的二阶偏导
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # 返回 PDE 残差: dy_t - 0.4 * dy_xx = 0
    return dy_t - 0.4 * dy_xx

# ==========================================
# 2. 定义解析解（真值公式：用作最终的裁判）
#    u(x,t) = e^(-0.4 * π² * t) * sin(π * x)
#    这是热传导方程在 Dirichlet 边界条件下的精确理论解
# ==========================================
def exact_solution(x):
    # x[:, 0:1] 是空间坐标 x，x[:, 1:2] 是时间坐标 t
    # 注意：用切片 0:1 和 1:2 而非 0 和 1，是为了保持二维数组形状 (N,1)
    return np.exp(-0.4 * (np.pi ** 2) * x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])

# ==========================================
# 2.5. Grubbs 异常值检验
#     目的：自动检测多次重复实验中因随机初始化导致的异常偏离值，
#     避免一个"跑飞"的实验毁掉整个配置的均值统计
# ==========================================
def grubbs_test(data):
    """
    Grubbs 单异常值检验（双尾，显著性水平固定为 0.05）。
    原理：G = max|xi - mean| / std，若 G > G_crit 则该点为异常值。
    因为 scipy 不一定已安装，这里硬编码 alpha=0.05 下的常用临界值。
    返回 (保留的索引列表, 被剔除的索引列表)
    """
    n = len(data)
    # 样本量不足 3 时，检验无统计学意义，全部保留
    if n < 3:
        return list(range(n)), []

    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 使用样本标准差（分母 n-1），与 Grubbs 定义一致

    # 如果所有值几乎相同，标准差为 0，不剔除任何值
    if std < 1e-15:
        return list(range(n)), []

    # 计算每个点的 Grubbs 统计量 G，取最大值
    G = np.abs(data - mean) / std
    max_idx = int(np.argmax(G))
    G_max = G[max_idx]

    # Grubbs 临界值表 (alpha=0.05, 双尾)
    # 计算公式: G_crit = (n-1)/sqrt(n) * sqrt(t² / (n-2 + t²))
    # 其中 t 是 t 分布 t_{1-alpha/(2n), n-2} 的上侧分位数
    g_table = {3: 1.155, 4: 1.481, 5: 1.715, 6: 1.887, 7: 2.020,
               8: 2.126, 9: 2.215, 10: 2.290, 15: 2.549, 20: 2.709}
    G_crit = g_table.get(n, 2.5)  # n 超出查表范围时取保守阈值

    if G_max > G_crit:
        keep = [i for i in range(n) if i != max_idx]
        removed = [max_idx]
        return keep, removed
    return list(range(n)), []

# ==========================================
# 3. 几何区域与定解条件设置
# ==========================================
# 空间区域：x ∈ [-1, 1]
geom = dde.geometry.Interval(-1, 1)
# 时间区域：t ∈ [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)
# 时空耦合区域：将空间和时间组合成一个二维问题域
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Dirichlet 边界条件：在空间边界上 u(-1,t) = u(1,t) = 0
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
# 初始条件：t=0 时 u(x,0) = sin(π * x)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 准备测试点：1000 个均匀分布的网格点用于最终评估
X_test = geomtime.uniform_points(1000, boundary=True)
y_true = exact_solution(X_test)

# ==========================================
# 4. 消融实验：多次重复 + 异常值剔除 + 训练发散检测
# ==========================================
# 每个配点数重复 5 次，让均值更可靠（原为 3 次）
domain_points_list = [100, 500, 2000]
num_repeats = 5                           # 每配置重复 5 次，降低随机初始化偶然性的影响
# 【改进1】迭代次数与配点数联动：配点越多 → 约束越多 → 需要更多训练步数
#         但配点多也意味着每步梯度信息更丰富，不需要线性放大
#         100→5000步, 500→10000步, 2000→20000步（次线性缩放，兼顾充分训练与总耗时）
def iterations_for(n_points):
    schedule = {100: 5000, 500: 10000, 2000: 20000}
    return schedule.get(n_points, 5000)

# 存储原始均值与标准差（剔除异常值前）
errors_l2_mean, errors_l2_std = [], []
times_mean, times_std = [], []
# 记录每轮被剔除的异常值数量
outlier_counts = []

print("开始自动消融实验（每个配置重复 5 次，含 Grubbs 异常值检验）...")
print(f"总训练轮数: {len(domain_points_list) * num_repeats}")

for n_points in domain_points_list:
    l2_runs = []      # 存储本轮所有 L2 Error（用于后续异常值检验）
    time_runs = []    # 存储本轮所有耗时

    for run_idx in range(1, num_repeats + 1):
        print(f"\n--- 配点={n_points}, 第 {run_idx}/{num_repeats} 次 ---")

        # 根据当前配点数量动态创建 PDE 数据集
        data = dde.data.TimePDE(
            geomtime, pde, [bc, ic],
            num_domain=n_points,
            num_boundary=40,
            num_initial=20,
            solution=exact_solution
        )

        # 重新初始化网络（消除不同配点量之间的权重干扰）
        net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        # 加入学习率逆时衰减：每 2000 步学习率减半，让后期训练更稳定
        # 【改进2】学习率策略从 "inverse time" 改为 "step"（阶梯衰减）
        #         inverse time 衰减过快导致后期训练不稳定、发散率高；
        #         step 每 2000 步减半，更可控
        model.compile("adam", lr=1e-3, decay=("step", 2000, 0.5))

        start_time = time.time()
        # 【改进3】接收 train() 返回的 losshistory 用于分析训练是否发散
        #         迭代次数按配点量动态缩放，2000 配点有 15000 步充分训练
        iters = iterations_for(n_points)
        losshistory, train_state = model.train(iterations=iters, display_every=2000)
        end_time = time.time()

        cost_time = end_time - start_time
        time_runs.append(cost_time)

        # 评价指标计算
        y_pred = model.predict(X_test)
        l2_error = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        max_error = np.max(np.abs(y_true - y_pred))
        l2_runs.append(l2_error)

        # 【改进4】从 loss 历史中检测训练是否发散（过拟合/震荡）
        # losshistory.loss_test 每一行是 [pde_loss, bc_loss, ic_loss]，
        # 用三者的和作为总 loss 来判断最优步数
        test_loss_arr = np.array(losshistory.loss_test)           # shape: (记录次数, 3)
        total_loss_curve = np.sum(test_loss_arr, axis=1)          # 每步的总 loss
        best_idx = int(np.argmin(total_loss_curve))               # 总 loss 最小的位置
        best_step = losshistory.steps[best_idx]                   # 对应的迭代步数
        final_step = losshistory.steps[-1]                        # 最后一步
        # 如果最佳步比最后步早了超过 500 步，说明训练后期在发散
        diverged = (final_step - best_step) > 500
        diverge_flag = "[发散]" if diverged else "[稳定]"

        print(f"  本轮 L2={l2_error:.6f}, Max={max_error:.6f}, 耗时={cost_time:.1f}s, "
              f"best@step={best_step}/{final_step} {diverge_flag}")

    # 【改进5】对 L2 Error 做 Grubbs 异常值检验，剔除"跑飞"的坏结果
    keep_idx, removed_idx = grubbs_test(np.array(l2_runs))
    l2_clean = [l2_runs[i] for i in keep_idx]
    time_clean = [time_runs[i] for i in keep_idx]

    n_outliers = len(removed_idx)
    outlier_counts.append(n_outliers)

    # 若本次有剔除，打印被剔除的异常值详情
    if n_outliers > 0:
        for ri in removed_idx:
            print(f"  [Grubbs] 剔除第 {ri+1} 次异常值: L2={l2_runs[ri]:.6f} "
                  f"(样本均值={np.mean(l2_runs):.6f}, 样本std={np.std(l2_runs, ddof=1):.6f})")

    # 基于清洗后的数据计算均值与标准差
    errors_l2_mean.append(np.mean(l2_clean))
    errors_l2_std.append(np.std(l2_clean, ddof=1))
    times_mean.append(np.mean(time_clean))
    times_std.append(np.std(time_clean, ddof=1))

    print(f"  => 清洗后均值 (剔除{n_outliers}个异常): L2={errors_l2_mean[-1]:.6f} +/- {errors_l2_std[-1]:.6f}, "
          f"耗时={times_mean[-1]:.1f}s +/- {times_std[-1]:.1f}s")
    if n_outliers > 0:
        print(f"     (原始含异常均值: L2={np.mean(l2_runs):.6f} +/- {np.std(l2_runs, ddof=1):.6f})")

# ==========================================
# 5. 绘制带误差棒的双 Y 轴折线图
#    - 只开水平网格线（axis='y'），去掉纵向竖线
#    - 使用更鲜明的现代配色
#    - 每个数据点标注数值，便于阅读
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# ---- 左 Y 轴：L2 Error ----
color_l2 = '#E52B50'  # 深红，比 tab:red 更鲜明
ax1.set_xlabel('配点数量 (num_domain)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Relative L2 Error', color=color_l2, fontsize=13, fontweight='bold')
line1 = ax1.errorbar(
    domain_points_list, errors_l2_mean, yerr=errors_l2_std,
    marker='D', color=color_l2, linewidth=2.5, capsize=6,
    markersize=9, markeredgecolor='white', markeredgewidth=1,
    label='L2 Error (mean ± std, 已剔除异常值)'
)
ax1.tick_params(axis='y', labelcolor=color_l2, labelsize=11)
ax1.set_xticks(domain_points_list)
ax1.tick_params(axis='x', labelsize=11)

# 左轴每个点上方标注均值
for i, (x, y) in enumerate(zip(domain_points_list, errors_l2_mean)):
    ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0, 14), ha='center', fontsize=9, color=color_l2, fontweight='bold')

# ---- 右 Y 轴：训练耗时 ----
ax2 = ax1.twinx()
color_time = '#1D5E9E'  # 深蓝
ax2.set_ylabel('训练耗时 (秒)', color=color_time, fontsize=13, fontweight='bold')
line2 = ax2.errorbar(
    domain_points_list, times_mean, yerr=times_std,
    marker='s', color=color_time, linewidth=2.5, linestyle='--', capsize=6,
    markersize=9, markeredgecolor='white', markeredgewidth=1,
    label='Training Time (mean ± std)'
)
ax2.tick_params(axis='y', labelcolor=color_time, labelsize=11)

# 右轴每个点上方标注均值
for i, (x, y) in enumerate(zip(domain_points_list, times_mean)):
    ax2.annotate(f'{y:.0f}s', (x, y), textcoords="offset points",
                 xytext=(0, 14), ha='center', fontsize=9, color=color_time, fontweight='bold')

# ---- 图例：合并左右轴的两条线 ----
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9,
           edgecolor='gray', fancybox=True)

# ---- 标题与网格 ----
plt.title('物理约束采样点数量对精度与耗时的影响\n'
          '(含 Grubbs 异常值剔除 + 训练发散检测 + 迭代次数动态缩放)',
          fontsize=14, fontweight='bold', pad=15)
# 关键：只开水平网格线（axis='y'），去掉纵向竖线
ax1.grid(axis='y', linestyle=':', alpha=0.4, color='gray')
# 用浅色横线强调 L2=0 基准
ax1.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)

fig.tight_layout()
plt.savefig('Ablation_Study_Result.png', dpi=300, bbox_inches='tight')
print("\n实验全部完成！折线图已保存为 'Ablation_Study_Result.png'。")

# 保存数值结果到文本文件
with open('Ablation_Study_Data.txt', 'w', encoding='utf-8') as f:
    f.write("n_points\tL2_mean\tL2_std\ttime_mean\ttime_std\toutliers_removed\n")
    for i, n in enumerate(domain_points_list):
        f.write(f"{n}\t{errors_l2_mean[i]:.6f}\t{errors_l2_std[i]:.6f}\t"
                f"{times_mean[i]:.1f}\t{times_std[i]:.1f}\t{outlier_counts[i]}\n")

plt.show()