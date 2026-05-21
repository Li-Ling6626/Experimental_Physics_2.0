import numpy as np
import matplotlib.pyplot as plt

# 1. 設定參數
a = 1.0       # 晶格常數 (設為1方便作圖)
t = 1.0       # 跳躍參數 (Hopping parameter)
E0 = 0.0      # 參考能量

# 2. 建立 k 的陣列 (第一布里淵區: -pi/a 到 pi/a)
k = np.linspace(-np.pi/a, np.pi/a, 400)

# 3. 計算能帶 E(k)
E_k = E0 - 2 * t * np.cos(k * a)

# 4. 開始作圖
plt.figure(figsize=(8, 5))
plt.plot(k, E_k, 'b-', lw=2, label=r'$E(k) = E_0 - 2t \cos(ka)$')

# 標示 k = 0 (Band Bottom)
plt.plot(0, E0 - 2*t, 'ro', markersize=8)
plt.text(0, E0 - 2*t - 0.2, 'k = 0\n(Band Bottom)', ha='center', va='top', color='red', fontsize=12)

# 標示 k = pi/a (Band Top)
plt.plot(np.pi/a, E0 + 2*t, 'go', markersize=8)
plt.text(np.pi/a, E0 + 2*t + 0.2, 'k = $\pi/a$\n(Band Top)', ha='center', va='bottom', color='green', fontsize=12)

# 畫出對稱軸與邊界
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(np.pi/a, color='gray', linestyle=':', alpha=0.5, label='BZ Boundary')
plt.axvline(-np.pi/a, color='gray', linestyle=':', alpha=0.5)

# 圖表美化
plt.title('1D Tight-Binding Band', fontsize=16)
plt.xlabel('Wave vector $k$', fontsize=14)
plt.ylabel('Energy $E(k)$', fontsize=14)
plt.xticks([-np.pi/a, 0, np.pi/a], [r'$-\pi/a$', '0', r'$\pi/a$'], fontsize=12)
plt.xlim(-np.pi/a * 1.1, np.pi/a * 1.1)
plt.ylim(E0 - 2*t - 0.8, E0 + 2*t + 0.8)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()