import numpy as np
import matplotlib.pyplot as plt

# 頻率範圍：從 0.001 Hz 到 1000 Hz，取對數分佈
f = np.logspace(-3, 3, 1000)

# 設定要比較的兩個 Time Constant
tau_small = 0.01  # 10ms
tau_large = 1.0   # 1s

# 計算截止頻率 fc = 1 / (2 * pi * tau)
fc_small = 1 / (2 * np.pi * tau_small)
fc_large = 1 / (2 * np.pi * tau_large)

# 計算 ENBW = 1 / (4 * tau)
enbw_small = 1 / (4 * tau_small)
enbw_large = 1 / (4 * tau_large)

# 計算振幅響應 (Magnitude in dB)
mag_small = -10 * np.log10(1 + (f / fc_small)**2)
mag_large = -10 * np.log10(1 + (f / fc_large)**2)

# --- 繪圖 ---
plt.figure(figsize=(10, 6))

# 繪製頻譜曲線
plt.semilogx(f, mag_small, label=f'$\\tau = {tau_small}$ s (Fast, Noisy)', color='red', linewidth=2)
plt.semilogx(f, mag_large, label=f'$\\tau = {tau_large}$ s (Slow, Quiet)', color='blue', linewidth=2)

# 標示 -3dB 截止頻率點
plt.axhline(-3, color='gray', linestyle='--', alpha=0.7)
plt.axvline(fc_small, color='red', linestyle=':', alpha=0.7)
plt.axvline(fc_large, color='blue', linestyle=':', alpha=0.7)

# === 更新：在圖上同時標示截止頻率與 ENBW 數值 ===
# 利用 \n 換行，將 ENBW 顯示在 fc 下方
plt.annotate(f'$f_c \\approx {fc_small:.1f}$ Hz\nENBW $= {enbw_small:.1f}$ Hz', 
             xy=(fc_small, -3), 
             xytext=(fc_small * 1.5, 2), # 稍微往上移避免壓到線
             color='red', fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.annotate(f'$f_c \\approx {fc_large:.2f}$ Hz\nENBW $= {enbw_large:.2f}$ Hz', 
             xy=(fc_large, -3), 
             xytext=(fc_large * 1.5, -12), # 稍微往下移
             color='blue', fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='blue'))
# ==================================

# 圖表設定
plt.title('Low-Pass Filter Frequency Response (6 dB/oct)', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Magnitude (dB)', fontsize=12)
plt.ylim(-60, 15) # 稍微拉高 Y 軸上限，讓文字有空間顯示
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12, loc='lower left')

plt.tight_layout()
plt.show()