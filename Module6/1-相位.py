import numpy as np
import matplotlib.pyplot as plt

# 定義參數
a = 1  # 假設晶格常數為 1
n = np.arange(6)  # 6 個 sites: 0, 1, 2, 3, 4, 5
k_values = [0, np.pi/(2*a), np.pi/a]
k_labels = ['k = 0', 'k = $\pi/2a$', 'k = $\pi/a$']

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for i, k in enumerate(k_values):
    # 計算 Bloch phase 係數 cn
    cn = np.exp(1j * k * n * a)
    
    # 畫出實部代表波函數在實空間的振幅
    axs[i].plot(n, np.real(cn), marker='o', linestyle='-', color='b')
    axs[i].axhline(0, color='gray', linestyle='--')
    axs[i].set_ylabel('Re($c_n$)')
    axs[i].set_title(f'Bloch Phase Real Part at {k_labels[i]}')
    axs[i].set_ylim(-1.5, 1.5)

axs[-1].set_xlabel('Site index (n)')
plt.tight_layout()
plt.show()