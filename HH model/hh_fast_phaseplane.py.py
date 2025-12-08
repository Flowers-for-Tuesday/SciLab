import numpy as np
import matplotlib.pyplot as plt

# ============================
# HH parameters
# ============================
C = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.4

# external current
I_ext = 10.0

# ============================
# α, β functions (textbook version)
# ============================
def alpha_m(V):
    return 0.1*(25-V)/(np.exp((25-V)/10)-1)

def beta_m(V):
    return 4*np.exp(-V/18)

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))


# ============================
# Fast subsystem: h, n are frozen
# ============================
h_fixed = 0.6
n_fixed = 0.32

def dV_dt(V, m):
    INa = g_Na*(m**3)*h_fixed*(V - E_Na)
    IK  = g_K*(n_fixed**4)*(V - E_K)
    IL  = g_L*(V - E_L)
    return (I_ext - INa - IK - IL) / C

def dm_dt(V, m):
    return alpha_m(V)*(1 - m) - beta_m(V)*m


# ============================
# Generate phase plane grid
# ============================
V = np.linspace(-80, 60, 40)
m = np.linspace(0, 1, 40)
VV, mm = np.meshgrid(V, m)

dV = dV_dt(VV, mm)
dm = dm_dt(VV, mm)

# ============================
# Compute nullclines
# ============================
# dm/dt = 0 → m = m_inf(V)
m_nullcline = m_inf(V)

# dV/dt = 0 → solve for m
V_nullcline = []
for v in V:
    ms = np.linspace(0, 1, 400)
    vals = dV_dt(v, ms)
    idx = np.argmin(np.abs(vals))
    V_nullcline.append(ms[idx])


# ============================
# Plotting
# ============================
plt.figure(figsize=(7, 6))

# Vector field
plt.quiver(VV, mm, dV, dm, angles='xy')

# Nullclines
plt.plot(V, m_nullcline, 'r', linewidth=2, label="dm/dt = 0")
plt.plot(V, V_nullcline, 'b', linewidth=2, label="dV/dt = 0")

plt.xlabel("V (mV)")
plt.ylabel("m")
plt.title("Hodgkin–Huxley Fast Subsystem Phase Plane (V–m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
