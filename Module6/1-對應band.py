import numpy as np
import matplotlib.pyplot as plt

# 設定參數
a = 1.0       # 晶格常數
t = 1.0       # 跳躍積分 (hopping parameter)，決定 band 的寬度
E_0 = 0.0     # 參考能量中心，設為 0

# 定義 k 的範圍：從 -pi/a 到 pi/a (第一布里淵區 First Brillouin Zone)
k = np.linspace(-np.pi/a, np.pi/a, 500)

# 計算一維 tight-binding model 的能帶能量 E(k)
E_k = E_0 - 2 * t * np.cos(k * a)

# 要標記的特定 k 值 (對應練習題)
k_points = [0, np.pi/(2*a), np.pi/a]
E_points = [E_0 - 2*t*np.cos(k_val * a) for k_val in k_points]
labels = ['k = 0\n(In-phase,\nBand Bottom)', 
          'k = $\pi/2a$\n(Intermediate)', 
          'k = $\pi/a$\n(Out-of-phase,\nBand Top)']

# 開始作圖
plt.figure(figsize=(10, 6))
plt.plot(k, E_k, 'b-', linewidth=2, label='Energy Band $E(k) = E_0 - 2t \cos(ka)$')

# 標出那三個點
colors = ['red', 'green', 'orange']
for i in range(3):
    plt.plot(k_points[i], E_points[i], marker='o', markersize=10, color=colors[i])
    plt.annotate(labels[i], 
                 (k_points[i], E_points[i]), 
                 textcoords="offset points", 
                 xytext=(10, 10 if i != 0 else -40), 
                 ha='left', fontsize=12, color=colors[i])

# 畫出對稱軸與輔助線
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(np.pi/a, color='gray', linestyle=':', alpha=0.5, label='BZ Boundary')
plt.axvline(-np.pi/a, color='gray', linestyle=':', alpha=0.5)
plt.axhline(E_0, color='gray', linestyle='--', alpha=0.3)

# 圖表設定
plt.title('1D Tight-Binding Band Structure', fontsize=16)
plt.xlabel('Wave vector $k$', fontsize=14)
plt.ylabel('Energy $E(k)$', fontsize=14)
plt.xticks([-np.pi/a, 0, np.pi/a], ['$-\pi/a$', '0 ($\Gamma$)', '$\pi/a$'], fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/a * 1.1, np.pi/a * 1.1)

plt.tight_layout()
plt.show()