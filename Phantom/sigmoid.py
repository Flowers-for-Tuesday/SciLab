import numpy as np
import matplotlib.pyplot as plt

def plot_activation_functions():
    """绘制s1和s2的稳态激活函数"""
    
    V_range = np.linspace(-80, 20, 1000)
    
    # 稳态激活函数（公式8-9）
    s1_inf = 1.0 / (1.0 + np.exp((-40.0 - V_range) / 0.5))
    s2_inf = 1.0 / (1.0 + np.exp((-42.0 - V_range) / 0.4))
    
    # 与其他激活函数对比
    m_inf = 1.0 / (1.0 + np.exp((-22.0 - V_range) / 7.5))
    n_inf = 1.0 / (1.0 + np.exp((-9.0 - V_range) / 10.0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 所有激活函数
    axes[0].plot(V_range, m_inf, 'red', linewidth=2, label='m∞ (Ca²⁺)')
    axes[0].plot(V_range, n_inf, 'blue', linewidth=2, label='n∞ (K⁺)')
    axes[0].plot(V_range, s1_inf, 'orange', linewidth=2, label='s1∞')
    axes[0].plot(V_range, s2_inf, 'purple', linewidth=2, label='s2∞')
    axes[0].set_xlabel('Membrane Potential (mV)', fontsize=12)
    axes[0].set_ylabel('Steady-state Activation', fontsize=12)
    axes[0].set_title('Steady-state Activation Functions', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axvline(x=-50, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    
    # s1和s2的陡峭性对比
    axes[1].plot(V_range, s1_inf, 'orange', linewidth=3, label='s1∞: slope = 1/0.5 = 2')
    axes[1].plot(V_range, s2_inf, 'purple', linewidth=3, label='s2∞: slope = 1/0.4 = 2.5')
    axes[1].set_xlabel('Membrane Potential (mV)', fontsize=12)
    axes[1].set_ylabel('Steady-state Activation', fontsize=12)
    axes[1].set_title('Steepness Comparison: s1∞ vs s2∞', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 标记激活的50%点
    axes[1].axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    axes[1].axvline(x=-40, color='orange', linestyle=':', alpha=0.5)
    axes[1].axvline(x=-42, color='purple', linestyle=':', alpha=0.5)
    axes[1].text(-40, 0.55, 'V_{1/2} = -40 mV', color='orange', ha='center')
    axes[1].text(-42, 0.45, 'V_{1/2} = -42 mV', color='purple', ha='center')
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("关键观察：")
    print("1. s1和s2的激活函数非常陡峭（斜率大），是开关式的")
    print("2. s2比s1稍陡峭，且V_{1/2}更低（-42 vs -40 mV）")
    print("3. 这意味着s2在更负的电压下就会激活")
    print("4. 这种陡峭性使它们在不同电压范围内起不同作用")

plot_activation_functions()