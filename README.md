# 基于 PINNs 的一维热传导方程求解与多维对比消融实验分析

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![DeepXDE](https://img.shields.io/badge/DeepXDE-1.11+-blue.svg?style=flat-square)](https://deepxde.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg?style=flat-square)]()

## 📖 项目简介
本项目是基于科学机器学习（SciML）领域的核心算法——**物理信息神经网络（PINNs）**开展的算法实验与交叉研究。项目聚焦于经典的一维热传导方程（1D Heat Equation），通过将偏微分方程（PDEs）、初始条件（IC）和边界条件（BC）作为物理硬约束嵌入神经网络的损失函数中，实现小样本/无监督条件下的高精度物理场演化预测。

本项目不仅实现了基础的 PINN 求解器，还重点构建了包含**传统有限差分法（FDM）**与**纯数据驱动神经网络（纯 NN）**的对比基线，并通过严密的消融实验，量化证明物理约束对模型泛化能力和稳定性的实质性贡献。

---

##  研究团队与分工
* **指导教师：** 吴国成/李腊全
* **开发团队 (智能科学与技术 + 数学与应用数学)：**
    * **葛林锋：** 算法与开发（Conda 环境搭建、DeepXDE 底层实现、PINN 求解器与基线模型代码开发）。
    * **顾宇航：** 数理与实验设计（热方程物理推演、FDM 解析解推导、消融实验控制变量设计）。
    * **柏川洋：** 评估与可视化（高阶评价指标计算、时空热力图绘制、数据整理与研究报告撰写）。

---

##  研究目标与主要内容
1.  **明确物理方程与数学定义：** 求解方程 $u_t - \alpha u_{xx} = 0$，空间域 $x \in [-1, 1]$，时间域 $t \in [0, 1]$，并基于分离变量法定义绝对真值基准。
2.  **多基线模型构建：** 完整实现 FDM 传统数值求解器与去除物理约束的纯 NN 模型，作为性能评估的上下限对照。
3.  **深度的物理损失项消融实验（Ablation Study）：**
    * **剥离 PDE 残差项：** 观测模型在缺乏物理法则监督时的退化现象。
    * **残差配点（Collocation Points）密度实验：** 动态调整采样点数量（如 100, 500, 2000），分析其对训练收敛与精度的影响。

---

##  技术路线与依赖栈
* **计算引擎：** `PyTorch` (自动微分与网络反向传播)
* **科学计算库：** `DeepXDE` (加速几何时空域定义与复合损失函数解耦)
* **数据处理与可视化：** `NumPy`, `Matplotlib` (时空热力图与误差等高线绘制)
* **开发与运行环境：** Conda 虚拟环境依赖隔离，VSCode Remote SSH 远程服务器调试。

### 环境安装 (快速开始)
```bash
# 1. 克隆本仓库
git clone [https://github.com/your-username/PINN-Heat-Equation.git](https://github.com/your-username/PINN-Heat-Equation.git)
cd PINN-Heat-Equation

# 2. 创建并激活 Conda 虚拟环境
conda create -n pinn_env python=3.9 -y
conda activate pinn_env

# 3. 安装核心依赖
pip install torch torchvision torchaudio
pip install deepxde matplotlib numpy
