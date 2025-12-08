import numpy as np
import matplotlib.pyplot as plt

# =========================
# 模型参数
# =========================
C_m = 1.0      # 膜电容 μF/cm^2
g_Na = 120.0   # 最大Na电导
g_K  = 36.0    # 最大K电导
g_L  = 0.3     # 漏电导
E_Na = 50.0    # Na反转电位 mV
E_K  = -77.0   # K反转电位 mV
E_L  = -54.387 # 漏电流反转电位 mV

# 外加电流（脉冲）
def I_ext(t):
    return 10 if 5 < t < 15 else 0  # μA/cm²

# =========================
# α−β 速率函数
# =========================
def alpha_m(V): return 0.1*(V+40)/(1 - np.exp(-(V+40)/10))
def beta_m(V):  return 4.0*np.exp(-(V+65)/18)

def alpha_h(V): return 0.07*np.exp(-(V+65)/20)
def beta_h(V):  return 1/(1 + np.exp(-(V+35)/10))

def alpha_n(V): return 0.01*(V+55)/(1 - np.exp(-(V+55)/10))
def beta_n(V):  return 0.125*np.exp(-(V+65)/80)

# =========================
# 时间设置
# =========================
dt = 0.01
T  = 15  # ms
time = np.arange(0, T, dt)

# =========================
# 初始化变量
# =========================
V = -65.0
m = alpha_m(V)/(alpha_m(V) + beta_m(V))
h = alpha_h(V)/(alpha_h(V) + beta_h(V))
n = alpha_n(V)/(alpha_n(V) + beta_n(V))

Vs, ms, hs, ns = [], [], [], []

# =========================
# 数值积分（Euler法）
# =========================
for t in time:
    # 电导
    gNa_t = g_Na * m**3 * h
    gK_t  = g_K * n**4
    
    # 电流
    I_Na = gNa_t * (V - E_Na)
    I_K  = gK_t  * (V - E_K)
    I_L  = g_L   * (V - E_L)

    # 膜电位微分方程
    dVdt = - (I_Na + I_K + I_L) + I_ext(t)
    V += dt * dVdt / C_m

    # 门控变量动力学
    m += dt * (alpha_m(V)*(1-m) - beta_m(V)*m)
    h += dt * (alpha_h(V)*(1-h) - beta_h(V)*h)
    n += dt * (alpha_n(V)*(1-n) - beta_n(V)*n)

    # 保存结果
    Vs.append(V)
    ms.append(m)
    hs.append(h)
    ns.append(n)

# =========================
# 绘图
# =========================
plt.figure(figsize=(12,6))

# 膜电位
plt.subplot(2,1,1)
plt.plot(time, Vs)
plt.title("Hodgkin-Huxley model simulation")
plt.ylabel("Membrane Potential (mV)")

# 门控变量
plt.subplot(2,1,2)
plt.plot(time, ms, label='m (Na activation)')
plt.plot(time, hs, label='h (Na inactivation)')
plt.plot(time, ns, label='n (K activation)')
plt.xlabel("Time (ms)")
plt.ylabel("Gating variables")
plt.legend()

plt.tight_layout()
plt.show()
