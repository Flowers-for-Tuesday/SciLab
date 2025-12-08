import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# FitzHugh–Nagumo parameters
# -------------------------
a = 0.7
b = 0.8
epsilon = 0.08
I_ext = 0.5   # 外加电流，可自由修改

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

# ============================================================
# Animation: phase plane trajectory + moving point + v(t), w(t)
# ============================================================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
fig.tight_layout(pad=3)

# ----- Phase plane -----
ax1.set_title("Phase plane (v-w)")
ax1.set_xlabel("v")
ax1.set_ylabel("w")
ax1.plot(v, w, alpha=0.3)
point_pw, = ax1.plot([], [], 'ro', markersize=6)

# ----- v(t) -----
ax2.set_title("v(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("v")
ax2.plot(time, v, alpha=0.3)
point_v, = ax2.plot([], [], 'ro', markersize=6)

# ----- w(t) -----
ax3.set_title("w(t)")
ax3.set_xlabel("t")
ax3.set_ylabel("w")
ax3.plot(time, w, alpha=0.3)
point_w, = ax3.plot([], [], 'ro', markersize=6)


# ----- update function -----
def update(i):
    # phase plane
    point_pw.set_data([v[i]], [w[i]])

    # v(t)
    point_v.set_data([time[i]], [v[i]])

    # w(t)
    point_w.set_data([time[i]], [w[i]])

    return point_pw, point_v, point_w


ani = FuncAnimation(fig, update, frames=len(time), interval=10, blit=False)

plt.show()
