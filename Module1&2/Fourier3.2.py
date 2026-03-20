# 引入模組
import numpy as np
import matplotlib.pyplot as plt

# 定義函式用來生成一個二維點陣
def lattice_simulation(spacing=20):
    size = 512
    lattice = np.zeros((size, size))
    
    # 生成點陣 (模擬原子排列)
    for i in range(0, size, spacing):
        for j in range(0, size, spacing):
            lattice[i, j] = 1
            
    # 傅立葉轉換
    fft_lat = np.fft.fftshift(np.fft.fft2(lattice))
    reciprocal = np.abs(fft_lat)**2   # 計算傅立葉轉換，透過對複數值取絕對值並平方，獲得倒空間的強度分布
    
    #畫圖圖圖圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(lattice[200:300, 200:300], cmap='gray') # 放大觀察局部
    ax1.set_title(f"Real Space Lattice (a={spacing})") # ax1顯示狹縫形狀
    
    ax2.imshow(np.log(reciprocal + 1), cmap='viridis') # ax2顯示繞射圖樣
    ax2.set_title("Reciprocal Lattice (FFT)")
    plt.show()

# 比較兩種間距：觀察間距 a 與倒晶格點距的關係
lattice_simulation(spacing=15)
lattice_simulation(spacing=30)

###試試看
#實空間，a較小 點陣更密集，a較大 點陣更稀疏
#倒空間，a較小 倒晶格點之間的距離較大，a較大 倒晶格點之間的距離較小