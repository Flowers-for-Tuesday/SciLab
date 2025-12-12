"""
Simplified Phantom Burster Model for Pancreatic β-Cells
Basic model with adjustable parameters showing voltage and slow variable dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class SimplePhantomBurster:
    """
    Simple Phantom Burster Model with adjustable parameters
    
    Three bursting modes:
    1. Fast Bursting: g_s1 = 20 pS, period ~3 seconds
    2. Medium Bursting: g_s1 = 7 pS, period ~15 seconds  
    3. Slow Bursting: g_s1 = 3 pS, period ~90 seconds
    """
    
    def __init__(self):
        # Default parameters (from paper Table 1)
        self.params = {
            'C_m': 4524e-3,      # Membrane capacitance (pF -> nF)
            'g_Ca': 280e-3,      # Fast Ca²⁺ conductance (pS -> nS)
            'g_K': 1300e-3,      # Fast K⁺ conductance (pS -> nS)
            'g_L': 25e-3,        # Leak conductance (pS -> nS)
            'g_s1': 20e-3,       # Slow K⁺ conductance (pS -> nS)
            'g_s2': 32e-3,       # Very slow K⁺ conductance (pS -> nS)
            'V_Ca': 100.0,       # Ca²⁺ reversal (mV)
            'V_K': -80.0,        # K⁺ reversal (mV)
            'V_L': -40.0,        # Leak reversal (mV)
            'tau_s1': 1000.0,    # s1 time constant (ms)
            'tau_s2': 120000.0,  # s2 time constant (ms, 2 min = 120000 ms)
        }
        
    # Steady-state activation functions
    def m_inf(self, V):
        """Ca²⁺ current activation"""
        return 1.0 / (1.0 + np.exp((-22.0 - V) / 7.5))
    
    def n_inf(self, V):
        """Fast K⁺ current activation"""
        return 1.0 / (1.0 + np.exp((-9.0 - V) / 10.0))
    
    def s1_inf(self, V):
        """Slow K⁺ current activation"""
        return 1.0 / (1.0 + np.exp((-40.0 - V) / 0.5))
    
    def s2_inf(self, V):
        """Very slow K⁺ current activation"""
        return 1.0 / (1.0 + np.exp((-42.0 - V) / 0.4))
    
    # Voltage-dependent time constant for n
    def tau_n_func(self, V):
        return 8.3 / (1.0 + np.exp((V + 9.0) / 10.0))
    
    # Differential equations
    def ode_system(self, t, y):
        V, n, s1, s2 = y
        
        # dV/dt
        m = self.m_inf(V)
        I_Ca = self.params['g_Ca'] * (m) * (V - self.params['V_Ca'])
        I_K = self.params['g_K'] * (n) * (V - self.params['V_K'])
        I_s1 = self.params['g_s1'] * s1 * (V - self.params['V_K'])
        I_s2 = self.params['g_s2'] * s2 * (V - self.params['V_K'])
        I_L = self.params['g_L'] * (V - self.params['V_L'])
        
        dV_dt = -(I_Ca + I_K + I_s1 + I_s2 + I_L) / self.params['C_m']
        
        # dn/dt
        tau_n = self.tau_n_func(V)
        dn_dt = (self.n_inf(V) - n) / tau_n
        
        # ds1/dt
        ds1_dt = (self.s1_inf(V) - s1) / self.params['tau_s1']
        
        # ds2/dt
        ds2_dt = (self.s2_inf(V) - s2) / self.params['tau_s2']
        
        return [dV_dt, dn_dt, ds1_dt, ds2_dt]
    
    def simulate(self, g_s1_value, simulation_time_ms, initial_conditions=None):
        """
        Run simulation
        
        Parameters:
        g_s1_value: value of g_s1 (pS)
        simulation_time_ms: simulation time (ms)
        initial_conditions: [V0, n0, s1_0, s2_0]
        """
        # Update g_s1 parameter
        self.params['g_s1'] = g_s1_value * 1e-3  # Convert pS to nS
        
        # Default initial conditions
        if initial_conditions is None:
            initial_conditions = [-60.0, 0.01, 0.05, 0.43]
        
        # Time points
        t_eval = np.linspace(0, simulation_time_ms, int(simulation_time_ms/0.1) + 1)
        
        # Solve ODEs
        sol = solve_ivp(
            self.ode_system,
            [0, simulation_time_ms],
            initial_conditions,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        return sol.t, sol.y

def plot_all_variables(t, y, g_s1_value, mode_name):
    """
    Plot all four state variables: V, n, s1, s2
    
    Parameters:
    t: time array (ms)
    y: state variables [V, n, s1, s2]
    g_s1_value: g_s1 value (pS)
    mode_name: name of bursting mode
    """
    V, n, s1, s2 = y
    t_sec = t / 1000.0  # Convert ms to seconds
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # Plot 1: Membrane potential (V)
    axes[0].plot(t_sec, V, 'black', linewidth=1.5)
    axes[0].set_ylabel('Membrane Potential (mV)', fontsize=12)
    axes[0].set_title(f'{mode_name} - g_s1 = {g_s1_value} pS', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=-50, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    axes[0].legend(loc='upper right')
    
    # Plot 2: Fast K⁺ activation variable (n)
    axes[1].plot(t_sec, n, 'blue', linewidth=2, label='n')
    axes[1].plot(t_sec, n**4, 'cyan', linewidth=2, alpha=0.7, label='n⁴ (used in I_K)')
    axes[1].set_ylabel('Fast K⁺ Activation (n)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='upper right')
    
    # Plot 3: Slow variable s1
    axes[2].plot(t_sec, s1, 'orange', linewidth=2)
    axes[2].set_ylabel('Slow Variable s1', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    # Plot 4: Very slow variable s2
    axes[3].plot(t_sec, s2, 'purple', linewidth=2)
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].set_ylabel('Very Slow Variable s2', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([0.4, 0.8])
    
    plt.tight_layout()
    return fig

def run_fast_bursting():
    """Run fast bursting simulation (g_s1 = 20 pS)"""
    print("Running Fast Bursting Simulation (g_s1 = 20 pS)...")
    
    model = SimplePhantomBurster()
    t, y = model.simulate(
        g_s1_value=20,  # pS
        simulation_time_ms=30000  # 30 seconds = 30000 ms
    )
    
    # Analyze bursting
    V = y[0]
    threshold = -50.0
    above_threshold = V > threshold
    crossings = np.where(np.diff(above_threshold.astype(int)))[0]
    
    if len(crossings) >= 2:
        burst_starts = crossings[::2]
        if len(burst_starts) > 1:
            periods = np.diff(t[burst_starts]) / 1000.0  # Convert to seconds
            avg_period = np.mean(periods)
            print(f"  Average burst period: {avg_period:.2f} seconds")
            print(f"  Number of bursts: {len(periods)+1}")
    
    # Plot all four variables
    fig = plot_all_variables(t, y, 20, 'Fast Bursting')
    plt.savefig('fast_bursting.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return t, y

def run_medium_bursting():
    """Run medium bursting simulation (g_s1 = 7 pS)"""
    print("\nRunning Medium Bursting Simulation (g_s1 = 7 pS)...")
    
    model = SimplePhantomBurster()
    t, y = model.simulate(
        g_s1_value=7,  # pS
        simulation_time_ms=120000  # 120 seconds = 120000 ms
    )
    
    # Analyze bursting
    V = y[0]
    threshold = -50.0
    above_threshold = V > threshold
    crossings = np.where(np.diff(above_threshold.astype(int)))[0]
    
    if len(crossings) >= 2:
        burst_starts = crossings[::2]
        if len(burst_starts) > 1:
            periods = np.diff(t[burst_starts]) / 1000.0  # Convert to seconds
            avg_period = np.mean(periods)
            print(f"  Average burst period: {avg_period:.2f} seconds")
            print(f"  Number of bursts: {len(periods)+1}")
    
    # Plot all four variables
    fig = plot_all_variables(t, y, 7, 'Medium Bursting')
    plt.savefig('medium_bursting.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return t, y

def run_slow_bursting():
    """Run slow bursting simulation (g_s1 = 3 pS)"""
    print("\nRunning Slow Bursting Simulation (g_s1 = 3 pS)...")
    
    model = SimplePhantomBurster()
    t, y = model.simulate(
        g_s1_value=3,  # pS
        simulation_time_ms=300000  # 300 seconds = 300000 ms
    )
    
    # Analyze bursting
    V = y[0]
    threshold = -50.0
    above_threshold = V > threshold
    crossings = np.where(np.diff(above_threshold.astype(int)))[0]
    
    if len(crossings) >= 2:
        burst_starts = crossings[::2]
        if len(burst_starts) > 1:
            periods = np.diff(t[burst_starts]) / 1000.0  # Convert to seconds
            avg_period = np.mean(periods)
            print(f"  Average burst period: {avg_period:.2f} seconds")
            print(f"  Number of bursts: {len(periods)+1}")
    
    # Plot all four variables
    fig = plot_all_variables(t, y, 3, 'Slow Bursting')
    plt.savefig('slow_bursting.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return t, y

def main():
    """Main function to run all simulations"""
    print("=" * 60)
    print("Simple Phantom Burster Model with n⁴ (HH-like)")
    print("Based on: Bertram et al. (2000), Biophysical Journal")
    print("I_K = g_K * n⁴ * (V - V_K) - Hodgkin-Huxley style")
    print("=" * 60)
    
    # Run predefined simulations
    print("\nRunning three bursting mode simulations...")
    
    # Fast bursting
    t_fast, y_fast = run_fast_bursting()
    
    # Medium bursting
    t_medium, y_medium = run_medium_bursting()
    
    # Slow bursting
    t_slow, y_slow = run_slow_bursting()
    
    print("\n" + "=" * 60)
    print("Simulations completed!")
    print("Images saved:")
    print("  - fast_bursting.png")
    print("  - medium_bursting.png")
    print("  - slow_bursting.png")
    print("\nEach image shows four variables:")
    print("  1. Membrane potential (V)")
    print("  2. Fast K⁺ activation (n and n⁴)")
    print("  3. Slow variable s1")
    print("  4. Very slow variable s2")
    print("=" * 60)

# Run the main function
if __name__ == "__main__":
    main()