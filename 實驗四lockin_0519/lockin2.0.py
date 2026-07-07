# 引入模組
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter 

# 模組一：訊號＆雜訊產生（記得按照講義給的數據 不客氣）
# 取樣設定
fs = 10000          # sampling rate
T = 15               # total time (s)
t = np.linspace(0, T, fs*T)

# 真實訊號
f_sig = 50        
omega = 2*np.pi*f_sig
A_sig = 10e-6 # 很小的訊號
signal = A_sig * np.sin(omega * t)
# 雜訊
noise_amp = 10e-3 #snr=10e-3
noise = noise_amp * np.random.randn(len(t))
# 實際量測
measured = signal + noise

# 畫高級雙軸圖
fig, ax1 = plt.subplots(figsize=(11, 5))
# 左側 Y 軸（真實信號）
color1 = '#1f77b4'  # 經典藍色
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Measured Voltage (V)', color=color1, fontsize=11)
line1 = ax1.plot(t, measured, color=color1, alpha=0.85, label='Measured (with noise)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.6)
# 建立右側 Y 軸 (共用 X 軸)
ax2 = ax1.twinx()  
# 繪製右側 Y 軸：真實訊號 (True Signal，放大顯示)
color2 = '#ff7f0e'  # 經典橘色
ax2.set_ylabel('True Signal Voltage (V)', color=color2, fontsize=11)
line2 = ax2.plot(t, signal, color=color2, linewidth=2.5, label='True Signal (Zoomed)')
ax2.tick_params(axis='y', labelcolor=color2)
# 動態調整右側 y 軸範圍，讓正弦波震盪比例更優雅 (保留上下一些留白)
ax2.set_ylim(-A_sig * 1.5, A_sig * 1.5)
# 關鍵修正：將 X 軸範圍限制在最後 0.1 秒 (0.1s - 0.2s)
ax1.set_xlim(0.1, 0.2)
# 合併左右兩側的圖例 (Legend) 
# 因為是雙軸，若分開呼叫 legend() 會重疊，必須手動合併
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.title('Weak Signal Buried in Noise (Dual Y-Axes Visualization)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# 模組二：混頻
# reference signal
reference = np.sin(omega * t)

# 開混
mixed = measured * reference

# 畫畫畫畫畫圖
plt.figure(figsize=(10,4))
plt.plot(t[:2000], mixed[:2000])
plt.title('Mixed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(alpha=0.3)
plt.show()

'''
他媽的就是這裡開始出問題
實驗課做了四小時還是畫不出來 心死 
'''
# 模組三：低通濾波(RC)
# 設定時間常數
tau = 0.5 # 為什麼最大只能0.5啊淦（管他的 能跑的都是好碼）
fc = 1 / (2*np.pi*tau) # 計算截止頻率

# 建立 filter 函數
def lowpass(data, fc, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = fc / nyquist # 正規化截止頻率
    b, a = butter(order, normal_cutoff, btype='low')
    # 計算濾波器在直流 (f=0) 處的實際增益：H(0) = sum(b) / sum(a)
    dc_gain = np.sum(b) / np.sum(a)
    
    # 關鍵修正 1：改用因果單向濾波器 lfilter，這才能模擬真實物理電路的延遲與響應
    filtered = lfilter(b, a, data)
    
    # 修正增益：除以 dc_gain 的平方（因為 filtfilt 濾了兩次，增益為 dc_gain^2）
    # 這樣可以確保不論濾波器階數為何，直流增益都精確為 1.0
    # 關鍵修正 2：因為 lfilter 只濾了一次，修正直流增益時只需除以 dc_gain 即可 (非 dc_gain 的平方)
    filtered_normalized = filtered / dc_gain
    return filtered_normalized

# 開濾！
filtered_6dB = lowpass(mixed, fc, fs, order=1)
filtered_24dB = lowpass(mixed, fc, fs, order=4)

# 找到 99% 收斂時間
# 計算最終值
final_6 = np.mean(filtered_6dB[-1000:])
final_24 = np.mean(filtered_24dB[-1000:])
# 由於做過增益歸一化，此處 final_6 與 final_24 應極為接近，我們取其平均作為共同的最終穩定值
final_shared = (final_6 + final_24) / 2
# 99%
target_6 = 0.99 * final_shared
target_24 = 0.99 * final_shared

# 找到第一次到達 99% 的時間
idx_6 = np.where(filtered_6dB >= target_6)[0][0]
idx_24 = np.where(filtered_24dB >= target_24)[0][0]

t6 = t[idx_6]
t24 = t[idx_24]

plt.figure(figsize=(12,6))

# 曲線
plt.plot(t, filtered_6dB, label=f'6 dB/oct  ({t6/tau:.1f}τ)', linewidth=2)
plt.plot(t, filtered_24dB, label=f'24 dB/oct ({t24/tau:.1f}τ)', linewidth=2)

# 最終值虛線
plt.axhline(final_shared, linestyle='--', color='gray', alpha=0.7, label=f'Steady State ({final_shared:.5f})')

# 99% 收斂位置
plt.axvline(t6, linestyle=':', alpha=0.7)
plt.axvline(t24, linestyle=':', alpha=0.7)

# 標記點
# 標記 99% 收斂點
plt.scatter(t6, filtered_6dB[idx_6], s=100, color='#1f77b4', edgecolors='black', zorder=5)
plt.scatter(t24, filtered_24dB[idx_24], s=100, color='#ff7f0e', edgecolors='black', zorder=5)

# 圖形設定
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('Lock-in Output', fontsize=13)
plt.title('Lock-in Amplifier Settling Time', fontsize=15)
plt.grid(alpha=0.3)
plt.legend()
plt.show()