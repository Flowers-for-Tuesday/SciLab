import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# FitzHugh–Nagumo parameters
# -------------------------
a = 0.7
b = 0.8
epsilon = 0.08
I_ext = 0.05   # 外加电流，可自由修改

# -------------------------
# FHN dynamics
# -------------------------
def fhn_rhs(state, I=I_ext):
    v, w = state
    dv = v - (v**3) / 3 - w + I
    dw = epsilon * (v + a - b * w)
    return np.array([dv, dw])

# -------------------------
# RK4 step
# -------------------------
def rk4_step(state, dt, rhs):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# -------------------------
# Simulation settings
# -------------------------
dt = 0.01
T = 400
time = np.arange(0, T, dt)

state = np.array([-1.0, 1.0])
traj = np.zeros((len(time), 2))
traj[0] = state

# -------------------------
# Integrate ODE
# -------------------------
for i in range(1, len(time)):
    state = rk4_step(state, dt, lambda s: fhn_rhs(s, I_ext))
    traj[i] = state

v = traj[:, 0]
w = traj[:, 1]

# -------------------------
# Plot 1: time series
# -------------------------
plt.figure(figsize=(10,4))
plt.plot(time, v, label="v (fast)")
plt.plot(time, w, label="w (slow)")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.title("FitzHugh–Nagumo model time series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Plot 2: Phase flow (phase portrait)
# -------------------------
V = np.linspace(-3, 3, 150)
W = np.linspace(-2, 3, 150)
VV, WW = np.meshgrid(V, W)

dV = VV - (VV**3)/3 - WW + I_ext
dW = epsilon * (VV + a - b*WW)

plt.figure(figsize=(7,7))

# --- 相流线（流函数图） ---
plt.streamplot(
    V, W, dV, dW,
    density=1.2,
    color=np.sqrt(dV**2 + dW**2),
    linewidth=1,
    arrowsize=1.2,
    cmap="viridis"
)

# --- nullclines ---
plt.plot(V, V - V**3/3 + I_ext, 'r', label="dv/dt = 0")
plt.plot(V, (V + a) / b, 'b', label="dw/dt = 0")

# --- 数值轨迹 ---
plt.plot(v, w, "k", linewidth=1.6, label="Trajectory")

plt.xlabel("v")
plt.ylabel("w")
plt.legend()
plt.title("FHN Phase Flow (Phase Portrait)")
plt.grid(True)
plt.tight_layout()
plt.show()
