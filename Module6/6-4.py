import subprocess
import sys

def ensure_package(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pip_name or import_name
        ])

# Auto-install
ensure_package("numpy")
ensure_package("matplotlib")

import matplotlib
matplotlib.use("TkAgg")  # important for VSCode normal .py execution

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

kB = 8.617333262e-5  # eV/K


def fermi(E, EF=0.0, T=300):
    if T < 1e-9:
        return (E < EF).astype(float)
    x = (E - EF) / (kB * T)
    x = np.clip(x, -700, 700)  # avoid overflow
    return 1.0 / (np.exp(x) + 1.0)


def lorentzian(E, E0, Gamma):
    return Gamma / ((E - E0) ** 2 + Gamma ** 2)


def frac_pi_label(m_selected, N):
    num = 2 * m_selected
    den = N

    if num == 0:
        return "0"

    g = np.gcd(abs(num), abs(den))
    num //= g
    den //= g

    sign = "-" if num < 0 else ""
    num_abs = abs(num)

    if den == 1:
        if num_abs == 1:
            return f"{sign}π/a"
        return f"{sign}{num_abs}π/a"
    else:
        if num_abs == 1:
            return f"{sign}π/{den}a"
        return f"{sign}{num_abs}π/{den}a"


def compute_all(state_index=6, N=12, t=1.0, Gamma=0.08, T=100.0, EF=0.0):
    a = 1.0

    m_values = np.arange(-N // 2, N // 2)
    k_values = 2 * np.pi * m_values / (N * a)
    E_values = -2 * t * np.cos(k_values * a)

    state_index = int(np.clip(state_index, 0, len(m_values) - 1))

    m_selected = m_values[state_index]
    k0 = k_values[state_index]
    E0 = E_values[state_index]

    k_pi_text = frac_pi_label(m_selected, N)
    k_decimal_text = f"{k0:.3f}"

    # Bloch phase on 6 sites
    sites = np.arange(6)
    R = sites * a
    coeff = np.exp(1j * k0 * R)

    # ARPES map
    k_dense = np.linspace(-np.pi / a, np.pi / a, 500)
    E_axis = np.linspace(-2.8 * abs(t), 2.8 * abs(t), 350)
    I = np.zeros((len(E_axis), len(k_dense)))

    sigma_k = 0.10
    occ_values = fermi(E_values, EF=EF, T=T)

    for i, kd in enumerate(k_dense):
        intensity_k = np.zeros_like(E_axis)
        for j in range(len(k_values)):
            k_weight = np.exp(-((kd - k_values[j]) ** 2) / (2 * sigma_k ** 2))
            intensity_k += lorentzian(E_axis, E_values[j], Gamma) * k_weight * occ_values[j]
        I[:, i] = intensity_k

    return {
        "a": a,
        "m_values": m_values,
        "k_values": k_values,
        "E_values": E_values,
        "m_selected": m_selected,
        "k0": k0,
        "E0": E0,
        "k_pi_text": k_pi_text,
        "k_decimal_text": k_decimal_text,
        "coeff": coeff,
        "sites": sites,
        "k_dense": k_dense,
        "E_axis": E_axis,
        "I": I,
        "EF": EF,
    }


# Initial parameters
params = {
    "state_index": 6,
    "N": 12,
    "t": 1.0,
    "Gamma": 0.08,
    "T": 100.0,
    "EF": 0.0,
}

data = compute_all(**params)

# Figure layout: 2x2, bottom spans both columns
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

ax1 = fig.add_subplot(gs[0, 0])   # Bloch phase
ax2 = fig.add_subplot(gs[0, 1])   # discrete E-k
ax3 = fig.add_subplot(gs[1, :])   # ARPES map

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.25, hspace=0.35, wspace=0.28)

# ----- Panel 1 -----
ax1.axhline(0, color='gray', lw=1)
ax1.axvline(0, color='gray', lw=1)

coeff = data["coeff"]
phase_points = ax1.scatter(np.real(coeff), np.imag(coeff), s=70)
phase_arrows = []
phase_texts = []

for n in range(len(data["sites"])):
    arrow = ax1.arrow(
        0, 0,
        np.real(coeff[n]), np.imag(coeff[n]),
        head_width=0.05,
        length_includes_head=True
    )
    txt = ax1.text(np.real(coeff[n]) + 0.03, np.imag(coeff[n]) + 0.03, f"{n}")
    phase_arrows.append(arrow)
    phase_texts.append(txt)

ax1.set_title(
    "Bloch phase on 6 sites\n"
    f"state index = {params['state_index']}, m = {data['m_selected']}, "
    f"k = {data['k_pi_text']}\n"
    f"(decimal: k = {data['k_decimal_text']})"
)
ax1.set_xlabel("Re(c_n)")
ax1.set_ylabel("Im(c_n)")
ax1.set_aspect('equal')
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)

# ----- Panel 2 -----
ek_points = ax2.scatter(data["k_values"], data["E_values"], s=35, label="allowed k states")
ek_line, = ax2.plot(data["k_values"], data["E_values"], '--', alpha=0.45)
ek_selected = ax2.scatter([data["k0"]], [data["E0"]], s=100, zorder=5, label="selected state")

ax2.set_title(
    "Discrete 1D tight-binding states\n"
    f"selected: m = {data['m_selected']}, k = {data['k_pi_text']}"
)
ax2.set_xlabel("k")
ax2.set_ylabel("E(k)")
ax2.set_xlim(-np.pi, np.pi)
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi/a$", r"$0$", r"$\pi/a$"])
ax2.legend(loc="best", fontsize=9)

# ----- Panel 3 -----
im = ax3.imshow(
    data["I"],
    extent=[data["k_dense"].min(), data["k_dense"].max(), data["E_axis"].min(), data["E_axis"].max()],
    aspect='auto',
    origin='lower'
)
arpes_points = ax3.scatter(data["k_values"], data["E_values"], s=18, label="discrete states")
fermi_line = ax3.axhline(data["EF"], ls='--', lw=1, label="E_F")

ax3.set_title("Simulated ARPES map from discrete k states")
ax3.set_xlabel("k")
ax3.set_ylabel("Energy")
ax3.set_xticks([-np.pi, 0, np.pi])
ax3.set_xticklabels([r"$-\pi/a$", r"$0$", r"$\pi/a$"])
ax3.legend(loc="best", fontsize=9)

cbar = plt.colorbar(im, ax=ax3, label="Intensity")

# ----- Sliders -----
ax_state = plt.axes([0.10, 0.16, 0.30, 0.03])
ax_N = plt.axes([0.58, 0.16, 0.25, 0.03])
ax_t = plt.axes([0.10, 0.11, 0.30, 0.03])
ax_Gamma = plt.axes([0.58, 0.11, 0.25, 0.03])
ax_T = plt.axes([0.10, 0.06, 0.30, 0.03])
ax_EF = plt.axes([0.58, 0.06, 0.25, 0.03])

slider_state = Slider(ax_state, "state", 0, 11, valinit=6, valstep=1)
slider_N = Slider(ax_N, "N", 4, 40, valinit=12, valstep=2)
slider_t = Slider(ax_t, "t", 0.3, 2.0, valinit=1.0, valstep=0.05)
slider_Gamma = Slider(ax_Gamma, "Gamma", 0.01, 0.30, valinit=0.08, valstep=0.01)
slider_T = Slider(ax_T, "T(K)", 1, 500, valinit=100.0, valstep=10)
slider_EF = Slider(ax_EF, "E_F", -1.0, 1.0, valinit=0.0, valstep=0.05)


def redraw_arrows_and_labels(coeff):
    global phase_arrows, phase_texts

    for artist in phase_arrows:
        artist.remove()
    for txt in phase_texts:
        txt.remove()

    phase_arrows = []
    phase_texts = []

    for n in range(6):
        arrow = ax1.arrow(
            0, 0,
            np.real(coeff[n]), np.imag(coeff[n]),
            head_width=0.05,
            length_includes_head=True
        )
        txt = ax1.text(np.real(coeff[n]) + 0.03, np.imag(coeff[n]) + 0.03, f"{n}")
        phase_arrows.append(arrow)
        phase_texts.append(txt)


def update(val):
    N_now = int(slider_N.val)
    state_now = int(slider_state.val)

    # keep state index valid after N changes
    max_state = N_now - 1
    if state_now > max_state:
        state_now = max_state

    new_params = {
        "state_index": state_now,
        "N": N_now,
        "t": slider_t.val,
        "Gamma": slider_Gamma.val,
        "T": slider_T.val,
        "EF": slider_EF.val,
    }

    d = compute_all(**new_params)

    # Panel 1 update
    phase_points.set_offsets(np.c_[np.real(d["coeff"]), np.imag(d["coeff"])])
    redraw_arrows_and_labels(d["coeff"])
    ax1.set_title(
        "Bloch phase on 6 sites\n"
        f"state index = {state_now}, m = {d['m_selected']}, "
        f"k = {d['k_pi_text']}\n"
        f"(decimal: k = {d['k_decimal_text']})"
    )

    # Panel 2 update
    ek_points.set_offsets(np.c_[d["k_values"], d["E_values"]])
    ek_line.set_data(d["k_values"], d["E_values"])
    ek_selected.set_offsets([[d["k0"], d["E0"]]])
    ax2.set_title(
        "Discrete 1D tight-binding states\n"
        f"selected: m = {d['m_selected']}, k = {d['k_pi_text']}"
    )

    # Panel 3 update
    im.set_data(d["I"])
    im.set_extent([d["k_dense"].min(), d["k_dense"].max(), d["E_axis"].min(), d["E_axis"].max()])
    arpes_points.set_offsets(np.c_[d["k_values"], d["E_values"]])

    # redraw Fermi line
    global fermi_line
    fermi_line.remove()
    fermi_line = ax3.axhline(d["EF"], ls='--', lw=1, label="E_F")

    # refresh legend
    leg = ax3.get_legend()
    if leg is not None:
        leg.remove()
    ax3.legend(loc="best", fontsize=9)

    fig.canvas.draw_idle()


slider_state.on_changed(update)
slider_N.on_changed(update)
slider_t.on_changed(update)
slider_Gamma.on_changed(update)
slider_T.on_changed(update)
slider_EF.on_changed(update)

plt.show()