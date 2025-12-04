
import numpy as np
import matplotlib.pyplot as plt

def calculate_center_of_mass(masses, positions):
    """
    Calculate center of mass using A-Level Further Maths concepts.
    
    Formula: r_cm = (Σ m_i * r_i) / Σ m_i
    where r is position vector (x, y)
    
    """
    total_mass = np.sum(masses)
    
    # Calculate x-coordinate of center of mass
    x_cm = np.sum(masses * positions[:, 0]) / total_mass
    
    # Calculate y-coordinate of center of mass
    y_cm = np.sum(masses * positions[:, 1]) / total_mass
    
    return np.array([x_cm, y_cm])

def calculate_moments(masses, positions, pivot_point=[0, 0]):
    """
    Calculate moment about a pivot point.
    
    A-Level Further Maths Concept: Moments
    Moment = Force × Perpendicular Distance
    
    For point masses: Moment = m × g × distance
    Since g is constant, we consider: Moment = m × distance
    """
    pivot = np.array(pivot_point)
    
    # Calculate position vectors relative to pivot
    relative_positions = positions - pivot
    
    # Calculate moment about pivot (taking moments about origin)
    # Moment in x-direction (about y-axis)
    moment_x = np.sum(masses * relative_positions[:, 0])
    
    # Moment in y-direction (about x-axis)
    moment_y = np.sum(masses * relative_positions[:, 1])
    
    return moment_x, moment_y

def check_equilibrium(masses, positions, pivot_point):
    """
    Check if system is in equilibrium about a pivot point.
    
    A-Level FM Concept: Equilibrium Conditions
    For equilibrium:
    1. Sum of forces = 0 (always satisfied for static system)
    2. Sum of moments = 0 (about any point)
    """
    moment_x, moment_y = calculate_moments(masses, positions, pivot_point)
    
    # System is in equilibrium if moments are close to zero
    tolerance = 1e-6
    is_equilibrium = abs(moment_x) < tolerance and abs(moment_y) < tolerance
    
    return is_equilibrium, moment_x, moment_y

def calculate_lamina_properties(masses, positions):
    """
    Calculate properties for a lamina (2D plane shape with mass).
    
    A-Level FM Concept: Laminas and Centre of Mass
    Used for finding centre of mass of composite shapes.
    """
    total_mass = np.sum(masses)
    center_of_mass = calculate_center_of_mass(masses, positions)
    
    # Calculate second moment of mass (related to moment of inertia)
    # I = Σ m_i × r_i² where r_i is distance from center of mass
    distances_squared = np.sum((positions - center_of_mass)**2, axis=1)
    second_moment = np.sum(masses * distances_squared)
    
    # Radius of gyration k where I = M × k²
    radius_of_gyration = np.sqrt(second_moment / total_mass)
    
    return second_moment, radius_of_gyration

def analyze_stability(center_of_mass, base_limits):
    """
    Analyze stability using A-Level mechanics principles.
    
    A-Level FM Concept: Stability and Equilibrium
    A system is stable if the vertical line through the center of mass
    passes through the base of support.
    """
    x_cm, y_cm = center_of_mass
    x_min, x_max, y_min, y_max = base_limits
    
    # Check if center of mass is within base limits
    stable_x = x_min <= x_cm <= x_max
    stable_y = y_min <= y_cm <= y_max
    
    if stable_x and stable_y:
        # Calculate margin of stability (distance to nearest edge)
        margin_x = min(x_cm - x_min, x_max - x_cm)
        margin_y = min(y_cm - y_min, y_max - y_cm)
        margin = min(margin_x, margin_y)
        
        return "Stable", margin
    else:
        # Calculate how far center of mass is outside base
        overhang_x = max(0, x_min - x_cm, x_cm - x_max)
        overhang_y = max(0, y_min - y_cm, y_cm - y_max)
        overhang = max(overhang_x, overhang_y)
        
        return "Unstable", overhang

def visualize_system(masses, positions, center_of_mass, base_limits):
    """Create visualization showing FM concepts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT PLOT: Physical representation
    colors = plt.cm.viridis(np.linspace(0, 1, len(masses)))
    
    for i in range(len(masses)):
        ax1.scatter(positions[i, 0], positions[i, 1], 
                   s=masses[i] * 300, 
                   color=colors[i],
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=2,
                   label=f'm_{i+1} = {masses[i]:.1f} kg at ({positions[i,0]:.1f}, {positions[i,1]:.1f})')
        
        # Draw position vector from origin
        ax1.arrow(0, 0, positions[i, 0], positions[i, 1],
                 head_width=0.3, head_length=0.2,
                 fc=colors[i], ec=colors[i], alpha=0.3,
                 linestyle='--', linewidth=1.5)
    
    # Plot center of mass
    ax1.scatter(center_of_mass[0], center_of_mass[1], 
               color='red', s=400, marker='X',
               edgecolors='darkred', linewidth=3,
               label=f'CM at ({center_of_mass[0]:.2f}, {center_of_mass[1]:.2f})', 
               zorder=10)
    
    # Draw position vector to center of mass
    ax1.arrow(0, 0, center_of_mass[0], center_of_mass[1],
             head_width=0.3, head_length=0.2,
             fc='red', ec='darkred', linewidth=2.5)
    
    # Plot base of support
    x_min, x_max, y_min, y_max = base_limits
    base_x = [x_min, x_max, x_max, x_min, x_min]
    base_y = [y_min, y_min, y_max, y_max, y_min]
    ax1.plot(base_x, base_y, 
            color='green', linestyle='--', linewidth=2.5,
            label='Base of Support')
    ax1.fill(base_x, base_y, color='green', alpha=0.1)
    
    # Draw vertical line through CM (to check stability)
    ax1.axvline(x=center_of_mass[0], color='red', linestyle=':', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('x position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('y position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Position Vectors and Centre of Mass', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.axis('equal')
    
    # RIGHT PLOT: Moment diagram
    # Calculate moments about origin
    moment_x, moment_y = calculate_moments(masses, positions, [0, 0])
    
    # Show moment arms
    for i in range(len(masses)):
        # Moment arm for x-direction (perpendicular distance from y-axis)
        ax2.plot([0, positions[i, 0]], [positions[i, 1], positions[i, 1]],
                'b--', alpha=0.5, linewidth=1.5)
        ax2.scatter(positions[i, 0], positions[i, 1], 
                   s=masses[i] * 300, color=colors[i],
                   alpha=0.6, edgecolors='black', linewidth=2)
        
        # Label with moment contribution
        moment_contrib = masses[i] * positions[i, 0]
        ax2.text(positions[i, 0], positions[i, 1] + 0.5, 
                f'M = {moment_contrib:.1f}', 
                fontsize=9, ha='center')
    
    # Mark pivot (origin)
    ax2.scatter(0, 0, s=500, marker='^', color='black',
               edgecolors='red', linewidth=3, label='Pivot (Origin)', zorder=10)
    
    ax2.set_xlabel('x position (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('y position (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Moment Arms about Origin', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Input the number of masses
n = int(input("\nEnter the number of point masses: "))

masses = np.zeros(n)
positions = np.zeros((n, 2))

print("\nEnter position vectors for each mass:")
for i in range(n):
    masses[i] = float(input(f"\nMass {i+1} (kg): "))
    positions[i, 0] = float(input(f"  x-coordinate (m): "))
    positions[i, 1] = float(input(f"  y-coordinate (m): "))

# Calculate center of mass
center_of_mass = calculate_center_of_mass(masses, positions)
total_mass = np.sum(masses)

# Calculate moments about origin
moment_x, moment_y = calculate_moments(masses, positions, [0, 0])

# Check if system is in equilibrium about center of mass
is_equilibrium, eq_moment_x, eq_moment_y = check_equilibrium(masses, positions, center_of_mass)

# Calculate lamina properties
second_moment, radius_of_gyration = calculate_lamina_properties(masses, positions)

# Get base of support for stability analysis
print("\n" + "-"*70)
print("Define the base of support (rectangular region):")
x_min = float(input("Minimum x-coordinate: "))
x_max = float(input("Maximum x-coordinate: "))
y_min = float(input("Minimum y-coordinate: "))
y_max = float(input("Maximum y-coordinate: "))
base_limits = [x_min, x_max, y_min, y_max]

stability_status, margin = analyze_stability(center_of_mass, base_limits)

print("\n1. BASIC PROPERTIES:")
print(f"   Total mass (M): {total_mass:.2f} kg")
print(f"   Position vector of CM: r_cm = {center_of_mass[0]:.3f}i + {center_of_mass[1]:.3f}j m")

print("\n2. MOMENTS ABOUT ORIGIN:")
print(f"   Moment about y-axis (M_y): {moment_x:.2f} kg·m")
print(f"   Moment about x-axis (M_x): {moment_y:.2f} kg·m")
print(f"   Total moment magnitude: {np.sqrt(moment_x**2 + moment_y**2):.2f} kg·m")

print("\n3. EQUILIBRIUM CHECK (about CM):")
if is_equilibrium:
    print("   ✓ System is in EQUILIBRIUM about the center of mass")
    print("   (Sum of moments = 0)")
else:
    print("   ✗ System is NOT in equilibrium about this point")
    print(f"   Moment about y-axis: {eq_moment_x:.4e} kg·m")
    print(f"   Moment about x-axis: {eq_moment_y:.4e} kg·m")

print("\n4. LAMINA PROPERTIES:")
print(f"   Second moment of mass (I): {second_moment:.2f} kg·m²")
print(f"   Radius of gyration (k): {radius_of_gyration:.3f} m")
print(f"   (Represents distribution of mass about CM)")

print("\n5. STABILITY ANALYSIS:")
print(f"   Status: {stability_status}")
if stability_status == "Stable":
    print(f"   Margin of stability: {margin:.3f} m")
    print(f"   (Distance from CM to nearest edge of base)")
else:
    print(f"   Overhang: {margin:.3f} m")
    print(f"   (Distance CM is outside the base)")

print("\n6. VERIFICATION (using moments principle):")
# Verify using principle: if moments about CM are zero, it's the correct CM
verify_x = np.sum(masses * (positions[:, 0] - center_of_mass[0]))
verify_y = np.sum(masses * (positions[:, 1] - center_of_mass[1]))
print(f"   Σm(x - x_cm) = {verify_x:.6f} ≈ 0 ✓")
print(f"   Σm(y - y_cm) = {verify_y:.6f} ≈ 0 ✓")

# Visualize
visualize_system(masses, positions, center_of_mass, base_limits)