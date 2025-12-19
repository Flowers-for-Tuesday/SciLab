import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# 记录开始时间
start_time = time.time()

size = 50
# 优化1: 使用向量化操作替代双重循环
np.random.seed(42)  # 设置随机种子以便复现结果
nsand = np.random.randint(0, 4, (size, size), dtype='uint8')

# 优化2: 预分配邻居偏移量
neighbors = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

def avalanch_fast(nsand, size):
    """
    Optimized avalanche function using NumPy operations for acceleration
    """
    # 找到所有需要崩塌的位置
    unstable_mask = nsand >= 4
    if not np.any(unstable_mask):
        return 0
    
    # 获取第一个不稳定位置
    unstable_idx = np.argwhere(unstable_mask)[0]
    i, j = unstable_idx
    
    # 执行崩塌
    nsand[i, j] -= 4
    
    # 优化邻居更新
    for di, dj in neighbors:
        ni, nj = i + di, j + dj
        if 0 <= ni < size and 0 <= nj < size:
            nsand[ni, nj] += 1
    
    return 1

# 优化3: 预分配结果数组（根据经验值调整大小）
nstep = 50000
magnitudes = np.zeros(nstep // 10, dtype=int)  # 预分配数组
mag_count = 0

# 优化4: 减少不必要的打印
for i in range(nstep):
    if i % 10000 == 0:  # 每10000步打印一次进度
        print(f"Progress: {i}/{nstep}")
    
    # 中心加沙
    #center = size // 2
    #nsand[center, center] += 1
    # 随机加沙
    i = np.random.randint(0, size)
    j = np.random.randint(0, size)
    nsand[i, j] += 1

    magnitude = 0
    
    # 循环直到没有雪崩
    while avalanch_fast(nsand, size) > 0:
        magnitude += 1
    
    if magnitude > 0:
        # 动态扩展数组（如果需要）
        if mag_count >= len(magnitudes):
            magnitudes = np.concatenate([magnitudes, np.zeros(len(magnitudes), dtype=int)])
        magnitudes[mag_count] = magnitude
        mag_count += 1

# 裁剪数组到实际大小
magnitudes = magnitudes[:mag_count]

# 计算频次分布
x = np.bincount(magnitudes)
valid_indices = np.where(x > 0)[0]
xdata = valid_indices
ydata = x[valid_indices]

# 转换为对数坐标
log_x = np.log10(xdata + 1e-10)  # 避免log(0)
log_y = np.log10(ydata + 1e-10)

# 创建图形
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

# 图0: magnitude vs event index (time/order)
ax0.plot(
    np.arange(1, len(magnitudes) + 1),
    magnitudes,
    lw=0.5,
    color='gray'
)
ax0.set_xlabel('Avalanche index')
ax0.set_ylabel('Avalanche magnitude')
ax0.set_title('Avalanche Magnitude Time Series')
ax0.grid(True, alpha=0.3)

# 图1: 原始数据和对数坐标
ax1.plot(ydata, xdata, 'bo', alpha=0.6, label='Raw data')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Avalanche size')
ax1.set_title('Power-law Distribution (Axes Swapped)')
ax1.grid(True, alpha=0.3)

# 图2: 对数坐标和拟合曲线
# 移除异常值（如果存在）
valid_mask = ~np.isnan(log_x) & ~np.isnan(log_y) & np.isfinite(log_x) & np.isfinite(log_y)
log_x_clean = log_x[valid_mask]
log_y_clean = log_y[valid_mask]

# 线性拟合（在log-log坐标中）
if len(log_x_clean) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x_clean, log_y_clean)
    
    # 生成拟合曲线
    fit_x = np.linspace(log_x_clean.min(), log_x_clean.max(), 100)
    fit_y = intercept + slope * fit_x
    
    ax2.plot(log_y_clean, log_x_clean, 'ro', alpha=0.6, label='Data points')
    ax2.plot(
        fit_y, fit_x, 'k-', linewidth=2,
        label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}'
    )
    
    # 显示幂律指数
    ax2.text(
        0.05, 0.95, f'Power-law exponent: {-slope:.3f}',
        transform=ax2.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
else:
    ax2.plot(log_y, log_x, 'ro', alpha=0.6, label='Data points')
    print("Warning: insufficient data for fitting")

ax2.set_xlabel(r'$\log_{10}$(Frequency)')
ax2.set_ylabel(r'$\log_{10}$(Avalanche size)')
ax2.set_title('Log–log Plot and Power-law Fit')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("powerlaw_with_fit.png", dpi=150, bbox_inches='tight')
plt.show()

# 性能统计
end_time = time.time()
print(f"\nPerformance statistics:")
print(f"Total runtime: {end_time - start_time:.2f} seconds")
print(f"Total number of avalanches: {len(magnitudes)}")
print(f"Maximum avalanche size: {np.max(magnitudes) if len(magnitudes) > 0 else 0}")
print(f"Average avalanche size: {np.mean(magnitudes) if len(magnitudes) > 0 else 0:.2f}")

# 可选: 保存数据以便进一步分析
np.savez(
    'sandpile_data.npz',
    magnitudes=magnitudes,
    nsand_final=nsand,
    log_x=log_x_clean if len(log_x_clean) > 1 else log_x,
    log_y=log_y_clean if len(log_y_clean) > 1 else log_y
)

# 可选: 可视化最终沙堆状态
if size <= 100:  # 只在尺寸较小时显示
    plt.figure(figsize=(8, 8))
    plt.imshow(nsand, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of grains')
    plt.title('Final sandpile configuration')
    plt.savefig("final_sandpile.png", dpi=150, bbox_inches='tight')
    plt.show()
