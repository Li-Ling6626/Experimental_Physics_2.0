import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# =========================
# 時間設定
# =========================

fs = 10000
T = 15

t = np.linspace(0, T, fs*T)

dt = t[1] - t[0]

# =========================
# step input
# =========================

step = np.ones_like(t)

# =========================
# time constant
# =========================

tau = 0.5

# =========================
# 一階 RC filter
# y[n] = y[n-1] + alpha(x-y)
# =========================

def rc_filter(x, tau, dt):

    alpha = dt / (tau + dt)

    y = np.zeros_like(x)

    for i in range(1, len(x)):

        y[i] = y[i-1] + alpha * (x[i] - y[i-1])

    return y

# =========================
# 6 dB/oct = 一次 RC
# =========================

y1 = rc_filter(step, tau, dt)

# =========================
# 24 dB/oct = 四次 RC cascade
# =========================

y4 = step.copy()

for _ in range(4):

    y4 = rc_filter(y4, tau, dt)

# =========================
# settling time
# =========================

def settling_time(y,
                   target=0.99,
                   hold_points=1000):

    within = y >= target

    for i in range(len(y)-hold_points):

        if np.all(within[i:i+hold_points]):

            return i

    return None

idx1 = settling_time(y1)

idx4 = settling_time(y4)

t1 = t[idx1]

t4 = t[idx4]

# =========================
# plot
# =========================

plt.figure(figsize=(12,6))

plt.plot(t,
         y1,
         linewidth=2,
         label=f'6 dB/oct ({t1/tau:.1f}τ)')

plt.plot(t,
         y4,
         linewidth=2,
         label=f'24 dB/oct ({t4/tau:.1f}τ)')

plt.axhline(0.99,
            linestyle='--',
            color='gray',
            label='99% level')

plt.axvline(t1,
            linestyle=':',
            alpha=0.7)

plt.axvline(t4,
            linestyle=':',
            alpha=0.7)

plt.xlabel('Time (s)', fontsize=13)

plt.ylabel('Normalized Output', fontsize=13)

plt.title('RC Cascade Step Response',
          fontsize=15)

plt.grid(alpha=0.3)

plt.legend()

plt.show()

# =========================
# print
# =========================

print('6 dB/oct')

print(f't = {t1:.3f} s')

print(f'≈ {t1/tau:.2f} τ')

print()

print('24 dB/oct')

print(f't = {t4:.3f} s')

print(f'≈ {t4/tau:.2f} τ')