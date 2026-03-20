# 引入模組
import numpy as np
import matplotlib.pyplot as plt

# 定義函式
# mode決定是單狹縫還是雙狹縫模式
# slit_dist:在雙狹縫模式下，兩條狹縫之間的距離
def slit_simulation(mode='single', slit_width=20, slit_dist=30):
    # 定義512x512的二維陣列模擬空間
    size = 512
    mask = np.zeros((size, size))
    center = size // 2    # 陣列的中心位置，用來定位狹縫
    
    if mode == 'single':
        # 單狹縫：在中心畫一條亮線 (高度 100, 寬度 slit_width)
        mask[center-50:center+50, center-slit_width//2 : center+slit_width//2] = 1
    elif mode == 'double':
        # 雙狹縫：兩條寬度相同、間距為 slit_dist 的亮線
        w = slit_width // 2
        d = slit_dist // 2
        mask[center-50:center+50, center-d-w : center-d+w] = 1
        mask[center-50:center+50, center+d-w : center+d+w] = 1

    # 傅立葉轉換
    fft_result = np.fft.fftshift(np.fft.fft2(mask))
    magnitude = np.abs(fft_result)**2   # 計算傅立葉轉換，透過對複數值取絕對值並平方，獲得每個頻率的光強度


    # 畫圖圖圖圖 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(mask, cmap='gray')   # ax1顯示狹縫形狀
    ax1.set_title(f"Aperture ({mode} slit)")
    
    # 使用 Log 縮放來增強繞射條紋的細節
    ax2.imshow(np.log(magnitude + 1), cmap='inferno')   # ax2顯示繞射圖樣
    ax2.set_title("Diffraction Pattern (FFT)")
    plt.show()

# ================= 測試區域 =================

# --- 1. 單狹縫比較：觀察寬度變窄時，繞射圖樣是否變寬 ---
slit_simulation(mode='single', slit_width=5)
slit_simulation(mode='single', slit_width=20)
# 比對這兩張圖的右半部 (FFT)，發現「窄狹縫 (5)」的繞射亮帶，比「寬狹縫 (20)」的亮帶還要寬廣
# 驗證了空間與頻率的反比關係

# --- 2. 雙狹縫比較：觀察干涉條紋與包絡線 ---
slit_simulation(mode='double', slit_width=5, slit_dist=30)
slit_simulation(mode='double', slit_width=15, slit_dist=30)
slit_simulation(mode='double', slit_width=15, slit_dist=60)
# 裡面的「細小垂直暗紋」間距是一樣的（因為狹縫間距 slit_dist 都是 30）。
# 但是整體亮帶的「外圍輪廓（包絡線）」變窄了（因為狹縫本身 slit_width 變寬了）。