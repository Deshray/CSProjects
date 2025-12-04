import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def simulate_pendulum(theta0, length, mass, g, damping_coeff, time_step, total_time):
    """
    Simulate pendulum motion using Euler method for nonlinear ODE with optional damping.
    
    ODE without damping: d²θ/dt² = -(g/L)sin(θ)
    ODE with damping: d²θ/dt² = -(g/L)sin(θ) - (b/m)ω
    
    Rewritten as two first-order ODEs:
        dθ/dt = ω
        dω/dt = -(g/L)sin(θ) - (b/m)ω
    
    where b is the damping coefficient (kg/s)
    """
    # Initialize time and angle arrays
    time = np.arange(0, total_time, time_step)
    theta = np.zeros_like(time)
    omega = np.zeros_like(time)  # Angular velocity
    
    # Initial conditions
    theta[0] = theta0  # initial angle (in radians)
    omega[0] = 0       # initial angular velocity
    
    # Energy arrays
    kinetic_energy = np.zeros_like(time)
    potential_energy = np.zeros_like(time)
    total_energy = np.zeros_like(time)
    energy_dissipated = np.zeros_like(time)
    
    # Calculate initial energy
    kinetic_energy[0] = 0.5 * mass * (length * omega[0])**2
    potential_energy[0] = mass * g * length * (1 - np.cos(theta[0]))
    total_energy[0] = kinetic_energy[0] + potential_energy[0]
    energy_dissipated[0] = 0

    # Simulate the motion using the Euler method
    for i in range(1, len(time)):
        # Calculate angular acceleration with damping
        # α = -(g/L)sin(θ) - (b/m)ω
        gravity_term = -(g / length) * np.sin(theta[i-1])
        damping_term = -(damping_coeff / mass) * omega[i-1]
        alpha = gravity_term + damping_term
        
        # Update angular velocity and angle (Euler method)
        omega[i] = omega[i-1] + alpha * time_step
        theta[i] = theta[i-1] + omega[i] * time_step
        
        # Calculate energies at each timestep
        kinetic_energy[i] = 0.5 * mass * (length * omega[i])**2
        potential_energy[i] = mass * g * length * (1 - np.cos(theta[i]))
        total_energy[i] = kinetic_energy[i] + potential_energy[i]
        
        # Calculate cumulative energy dissipated by damping
        energy_dissipated[i] = total_energy[0] - total_energy[i]
    
    return time, theta, omega, kinetic_energy, potential_energy, total_energy, energy_dissipated

def analyze_pendulum_motion(time, theta):
    """
    Analyze period and amplitude of pendulum oscillations using peak detection.
    """
    # Find peaks (maxima) in the angular displacement to determine the period
    peaks, properties = find_peaks(theta, height=0)
    
    if len(peaks) > 1:
        # Calculate periods as time differences between consecutive peaks
        periods = np.diff(time[peaks])
        average_period = np.mean(periods)
        
        # Calculate amplitude as mean of peak heights
        peak_heights = theta[peaks]
        average_amplitude = np.mean(np.abs(peak_heights))
        
        print(f"Average Period: {average_period:.4f} seconds")
        print(f"Average Amplitude: {np.degrees(average_amplitude):.4f} degrees")
        print(f"Number of complete oscillations: {len(peaks) - 1}")
        print(f"Frequency: {1/average_period:.4f} Hz")
        
        return average_period, average_amplitude
    else:
        print("\nNot enough peaks found to calculate period.")
        print("Try increasing the total simulation time or initial angle.\n")
        return None, None

def plot_motion(time, theta, omega):
    """
    Plot angular displacement and angular velocity over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot angular displacement
    ax1.plot(time, np.degrees(theta), 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Angular Displacement (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Pendulum Angular Displacement vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # Plot angular velocity
    ax2.plot(time, omega, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Pendulum Angular Velocity vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_energy(time, kinetic_energy, potential_energy, total_energy, energy_dissipated, has_damping):
    """
    Plot energy components over time to verify conservation of energy (or dissipation with damping).
    """
    plt.figure(figsize=(12, 7))
    
    plt.plot(time, kinetic_energy, label='Kinetic Energy', color='red', linewidth=2)
    plt.plot(time, potential_energy, label='Potential Energy', color='green', linewidth=2)
    plt.plot(time, total_energy, label='Total Mechanical Energy', color='blue', linewidth=2, linestyle='--')
    
    if has_damping:
        plt.plot(time, energy_dissipated, label='Energy Dissipated (Damping)', 
                color='orange', linewidth=2, linestyle=':')
    
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Energy (Joules)', fontsize=12, fontweight='bold')
    
    if has_damping:
        plt.title('Energy Dissipation in Damped Pendulum Motion', fontsize=14, fontweight='bold')
    else:
        plt.title('Energy Conservation in Undamped Pendulum Motion', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Calculate and display energy metrics
    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    energy_change = abs(final_energy - initial_energy)
    energy_change_percent = (energy_change / initial_energy) * 100
    
    textstr = f'Initial Energy: {initial_energy:.4f} J\n'
    textstr += f'Final Energy: {final_energy:.4f} J\n'
    textstr += f'Energy Change: {energy_change:.4f} J ({energy_change_percent:.2f}%)\n'
    
    if has_damping:
        textstr += f'Total Dissipated: {energy_dissipated[-1]:.4f} J'
    
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_phase_space(theta, omega):
    """
    Plot phase space diagram (theta vs omega).
    """
    plt.figure(figsize=(10, 8))
    plt.plot(np.degrees(theta), omega, 'b-', linewidth=1.5, alpha=0.7)
    plt.scatter(np.degrees(theta[0]), omega[0], color='green', s=100, 
                marker='o', label='Start', zorder=5)
    plt.scatter(np.degrees(theta[-1]), omega[-1], color='red', s=100, 
                marker='x', label='End', zorder=5)
    
    plt.xlabel('Angular Displacement (degrees)', fontsize=12, fontweight='bold')
    plt.ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
    plt.title('Phase Space Diagram (θ vs ω)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# Get user input
theta0_deg = float(input("Enter the initial angle (in degrees): "))
theta0 = np.radians(theta0_deg)

length = float(input("Enter the length of the pendulum (in meters): "))
mass = float(input("Enter the mass of the pendulum bob (in kg): "))

g = 9.81  # Acceleration due to gravity (m/s^2)

# Ask about damping
include_damping = input("\nInclude air resistance/damping? (yes/no): ").lower().strip()

if include_damping == 'yes':
    print("\nDamping coefficient (b) represents air resistance.")
    print("Typical values: 0.1-0.5 for light damping, 0.5-2.0 for moderate damping")
    damping_coeff = float(input("Enter damping coefficient b (kg/s, e.g., 0.2): "))
    has_damping = True
else:
    damping_coeff = 0.0
    has_damping = False

time_step = float(input("\nEnter the time step for the simulation (in seconds, e.g., 0.01): "))
total_time = float(input("Enter the total time for the simulation (in seconds): "))

# Simulate the pendulum
if has_damping:
    print(f"\nSimulating damped pendulum motion (b = {damping_coeff} kg/s)...")

time, theta, omega, kinetic_energy, potential_energy, total_energy, energy_dissipated = simulate_pendulum(
    theta0, length, mass, g, damping_coeff, time_step, total_time)

# Analyze the pendulum motion
analyze_pendulum_motion(time, theta)

# Generate all plots
print("Generating visualizations...")
plot_motion(time, theta, omega)
plot_energy(time, kinetic_energy, potential_energy, total_energy, energy_dissipated, has_damping)
plot_phase_space(theta, omega)

if has_damping:
    print(f"\nWith damping (b = {damping_coeff} kg/s):")
    print(f"  - Initial energy: {total_energy[0]:.4f} J")
    print(f"  - Final energy: {total_energy[-1]:.4f} J")
    print(f"  - Energy dissipated: {energy_dissipated[-1]:.4f} J")
    print(f"  - Energy remaining: {(total_energy[-1]/total_energy[0])*100:.2f}%")
else:
    print("\nWithout damping:")
    print(f"  - Energy should be conserved (numerical errors may cause small changes)")
    print(f"  - Energy change: {abs(total_energy[-1] - total_energy[0]):.6f} J")
