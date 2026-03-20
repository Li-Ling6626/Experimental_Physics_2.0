# 引入模組
import numpy as np
import matplotlib.pyplot as plt

#定義函式指定晶格類型
def create_lattice(lattice_type, grid_size=512, a=40, sigma=3.0):
    grid = np.zeros((grid_size, grid_size))   # 建立2D晶格陣列
    center = grid_size // 2
    
    # 定義基本晶格向量
    if lattice_type == 'square':   # 正方晶格向量
        a1 = np.array([a, 0])  #沿x軸的向量
        a2 = np.array([0, a])  #沿y軸的向量
    elif lattice_type == 'hexagonal':   # 六角晶格向量 
        a1 = np.array([a, 0])  #沿x軸的向量
        a2 = np.array([a/2, a * np.sqrt(3)/2])  #120度角的向量
        
    Y, X = np.ogrid[:grid_size, :grid_size]  #生成網格座標
    n_range = range(-15, 16)
    
    for i in n_range:
        for j in n_range:         
            pos = center + i * a1 + j * a2   # 透過晶格向量的線性組合，計算每個原子的中心座標
            x, y = pos[0], pos[1]
            
            if 0 <= x < grid_size and 0 <= y < grid_size:
                dist_sq = (X - x)**2 + (Y - y)**2   #計算每個網格點到原子中心的距離平方
                grid += np.exp(-dist_sq / (2 * sigma**2))   #使用高斯函數模擬原子的強度分布
                
    return grid, a1, a2, center

#進行傅立葉轉換
def plot_fft_with_vectors(real_space_grid, a1, a2, center, title):
    fft_result = np.fft.fft2(real_space_grid)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_shifted)**2 + 1)   #計算幅度並取對數 
    ##為了將傅立葉轉換複數部分轉換為可視化的強度分布
    ##可以更清晰觀察到倒空間的重要特質

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. 畫實空間
    ax1.imshow(real_space_grid, cmap='gray', origin='lower')
    ax1.set_title(f'\nReal Space: {title}')
    
    # 使用 arrow 畫出 a1 (紅色) 和 a2 (藍色)
    ax1.arrow(center, center, a1[0], a1[1], color='red', width=1.5, 
              head_width=6, length_includes_head=True, label=r'$\vec{a}_1$')
    ax1.arrow(center, center, a2[0], a2[1], color='cyan', width=1.5, 
              head_width=6, length_includes_head=True, label=r'$\vec{a}_2$')
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # 2. 畫倒空間
    ax2.imshow(magnitude_spectrum, cmap='inferno', origin='lower', 
               vmin=np.percentile(magnitude_spectrum, 90), 
               vmax=np.max(magnitude_spectrum))
    ax2.set_title(f'\nReciprocal Space (FFT): {title}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- 執行主程式 ---
# 1. 正方晶格模擬
grid_sq, a1_sq, a2_sq, center = create_lattice('square')
plot_fft_with_vectors(grid_sq, a1_sq, a2_sq, center, 'Square Lattice')

# 2. 六角晶格模擬
grid_hex, a1_hex, a2_hex, center = create_lattice('hexagonal')
plot_fft_with_vectors(grid_hex, a1_hex, a2_hex, center, 'Hexagonal Lattice')