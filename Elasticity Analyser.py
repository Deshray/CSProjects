import numpy as np
import matplotlib.pyplot as plt

def calculate_youngs_modulus(strain, stress):
    """
    Calculate Young's Modulus from the linear (elastic) region.
    
    A-Level Physics: E = stress / strain (in elastic region)
    
    Young's Modulus is the gradient of the linear portion of 
    the stress-strain graph (elastic region).
    """
    # Use first 10% of data for elastic region (Hooke's Law applies)
    linear_region = slice(0, int(len(strain) * 0.1))
    
    # Linear regression to find gradient
    E = np.polyfit(strain[linear_region], stress[linear_region], 1)[0]
    
    return E

def find_elastic_limit(strain, stress, E):
    """
    Find elastic limit where material stops obeying Hooke's Law.
    
    A-Level Physics: Beyond elastic limit, permanent deformation occurs.
    """
    # Calculate expected stress if Hooke's Law still applied
    linear_stress = E * strain
    
    # Find where actual stress deviates significantly
    deviation = np.abs(stress - linear_stress) / stress
    
    for i in range(1, len(deviation)):
        if deviation[i] > 0.05:  # 5% deviation from linearity
            return strain[i], stress[i]
    
    return strain[0], stress[0]

def find_yield_point(strain, stress):
    """
    Find yield point (where plastic deformation begins).
    
    A-Level Physics: Yield point marks transition from elastic to plastic.
    After yield point, material won't return to original shape.
    """
    # Find where gradient changes significantly
    gradients = np.diff(stress) / np.diff(strain)
    
    # Initial gradient (in elastic region)
    initial_gradient = np.mean(gradients[:int(len(gradients)*0.1)])
    
    # Find where gradient drops to 50% of initial
    for i in range(len(gradients)):
        if gradients[i] < 0.5 * initial_gradient:
            return strain[i], stress[i]
    
    return strain[int(len(strain)*0.2)], stress[int(len(stress)*0.2)]

def calculate_strain_energy(strain, stress):
    """
    Calculate strain energy (area under stress-strain curve).
    
    A-Level Physics: Strain energy = ∫ stress dε
    This is the work done in deforming the material.
    For elastic region: Energy = ½ × stress × strain (triangle area)
    """
    # Total energy (area under entire curve) - using trapezoidal rule
    total_energy = np.trapz(stress, strain)
    
    return total_energy

def calculate_elastic_energy(strain, stress, elastic_limit_index):
    """
    Calculate elastic strain energy stored before elastic limit.
    
    A-Level Physics: This energy can be recovered when load is removed.
    """
    elastic_energy = np.trapz(stress[:elastic_limit_index], 
                              strain[:elastic_limit_index])
    
    return elastic_energy

def analyze_ductility(max_strain):
    """
    Classify material as ductile or brittle.
    
    A-Level Physics:
    - Ductile: Can undergo large plastic deformation (e.g., copper)
    - Brittle: Breaks with little deformation (e.g., glass)
    """
    percentage_elongation = max_strain * 100
    
    if percentage_elongation > 5:
        classification = "DUCTILE"
        description = "Can be drawn into wires, undergoes plastic deformation"
    else:
        classification = "BRITTLE"
        description = "Breaks with little deformation, fractures suddenly"
    
    return classification, description, percentage_elongation

def visualize_stress_strain(strain, stress, E, elastic_point, yield_point, uts_point):
    """
    Create A-Level Physics style stress-strain diagram with annotations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MAIN PLOT - Full stress-strain curve
    ax1.plot(strain * 100, stress / 1e6, linewidth=3, color='#2E86AB', 
            label='Stress-Strain Curve')
    
    # Draw Hooke's Law region (elastic)
    elastic_end_index = int(len(strain) * 0.15)
    hookes_law_line = E * strain[:elastic_end_index]
    ax1.plot(strain[:elastic_end_index] * 100, hookes_law_line / 1e6, 
            'g--', linewidth=2.5, alpha=0.7,
            label=f"Hooke's Law Region (E = {E/1e9:.1f} GPa)")
    
    # Mark key points
    points = [
        (elastic_point, 'Elastic Limit', 'orange', 'o'),
        (yield_point, 'Yield Point', 'red', 's'),
        (uts_point, 'Breaking Point (UTS)', 'darkred', 'X')
    ]
    
    for point, label, color, marker in points:
        if point is not None:
            ax1.scatter(point[0] * 100, point[1] / 1e6, 
                       s=300, color=color, marker=marker,
                       edgecolors='black', linewidth=2,
                       label=label, zorder=5)
            ax1.axvline(x=point[0] * 100, color=color, linestyle=':', alpha=0.4)
    
    # Shade elastic region
    elastic_idx = np.argmin(np.abs(strain - elastic_point[0]))
    ax1.fill_between(strain[:elastic_idx] * 100, 
                     stress[:elastic_idx] / 1e6,
                     alpha=0.2, color='green',
                     label='Elastic Region (Reversible)')
    
    # Shade plastic region
    ax1.fill_between(strain[elastic_idx:] * 100, 
                     stress[elastic_idx:] / 1e6,
                     alpha=0.2, color='red',
                     label='Plastic Region (Permanent)')
    
    ax1.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
    ax1.set_title('A-Level Physics: Stress-Strain Curve with Key Features', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations
    ax1.annotate('LINEAR\n(Hooke\'s Law)', 
                xy=(strain[elastic_idx//2] * 100, stress[elastic_idx//2] / 1e6),
                fontsize=11, fontweight='bold', color='green',
                ha='center')
    
    # BOTTOM PLOT - Elastic region in detail (for Hooke's Law demonstration)
    elastic_detailed = int(len(strain) * 0.12)
    ax2.plot(strain[:elastic_detailed] * 100, stress[:elastic_detailed] / 1e6,
            'o-', linewidth=2, markersize=6, color='#2E86AB',
            label='Experimental Data')
    
    # Plot Hooke's Law fit
    hookes_fit = E * strain[:elastic_detailed]
    ax2.plot(strain[:elastic_detailed] * 100, hookes_fit / 1e6,
            'r--', linewidth=2.5, alpha=0.8,
            label=f'Hooke\'s Law: σ = Eε\nE = {E/1e9:.2f} GPa')
    
    # Add formula annotation
    ax2.text(0.05, 0.95, 
            'Hooke\'s Law: σ = Eε\nwhere:\nσ = stress\nε = strain\nE = Young modulus',
            transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
    ax2.set_title('Elastic Region (Hooke\'s Law) - Linear Relationship', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.show()

print("\n📝 Enter stress-strain data from tensile testing experiment")

# Check if user wants sample data
use_sample = input("\nUse sample data for steel? (yes/no): ").lower()

if use_sample == 'yes':
    # Sample data for mild steel (typical A-Level example)
    print("\nUsing sample data for Mild Steel...")
    strain = np.array([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 
                      0.012, 0.016, 0.020, 0.030, 0.050, 0.080, 0.120, 0.150])
    stress = np.array([0, 200e6, 400e6, 600e6, 800e6, 1000e6, 1100e6, 1200e6,
                      1250e6, 1280e6, 1300e6, 1320e6, 1350e6, 1380e6, 1400e6, 1350e6])
else:
    n = int(input("\nNumber of data points: "))
    strain = np.zeros(n)
    stress = np.zeros(n)
    
    print("\nTip: Strain is dimensionless (e.g., 0.01 = 1%)")
    print("     Stress in Pascals (e.g., 500e6 = 500 MPa)")
    
    for i in range(n):
        strain[i] = float(input(f"\nPoint {i+1} - Strain: "))
        stress[i] = float(input(f"Point {i+1} - Stress (Pa): "))

# Sort data by strain
sorted_indices = np.argsort(strain)
strain = strain[sorted_indices]
stress = stress[sorted_indices]

print(f"\n1. YOUNG'S MODULUS (E)")
print(f"   E = stress/strain (in elastic region)")
print(f"   E = {E:.2e} Pa = {E/1e9:.2f} GPa")
print(f"   (This measures the stiffness of the material)")

# 2. Find elastic limit
elastic_strain, elastic_stress = find_elastic_limit(strain, stress, E)
elastic_idx = np.argmin(np.abs(strain - elastic_strain))
print(f"\n2. ELASTIC LIMIT")
print(f"   Strain: {elastic_strain:.4f} ({elastic_strain*100:.2f}%)")
print(f"   Stress: {elastic_stress:.2e} Pa ({elastic_stress/1e6:.1f} MPa)")
print(f"   (Beyond this point, deformation becomes permanent)")

# 3. Find yield point
yield_strain, yield_stress = find_yield_point(strain, stress)
print(f"\n3. YIELD POINT")
print(f"   Strain: {yield_strain:.4f} ({yield_strain*100:.2f}%)")
print(f"   Stress: {yield_stress:.2e} Pa ({yield_stress/1e6:.1f} MPa)")
print(f"   (Material undergoes plastic deformation beyond this)")

# 4. Ultimate Tensile Strength (Breaking Point)
uts_idx = np.argmax(stress)
uts_strain = strain[uts_idx]
uts_stress = stress[uts_idx]
print(f"\n4. ULTIMATE TENSILE STRENGTH (UTS)")
print(f"   Maximum stress: {uts_stress:.2e} Pa ({uts_stress/1e6:.1f} MPa)")
print(f"   Occurs at strain: {uts_strain:.4f} ({uts_strain*100:.2f}%)")
print(f"   (Maximum stress material can withstand before necking/breaking)")

# 5. Strain Energy
total_energy = calculate_strain_energy(strain, stress)
elastic_energy = calculate_elastic_energy(strain, stress, elastic_idx)
plastic_energy = total_energy - elastic_energy

print(f"\n5. STRAIN ENERGY (Work Done)")
print(f"   Total energy: {total_energy:.2e} J/m³ ({total_energy/1e6:.2f} MJ/m³)")
print(f"   Elastic energy: {elastic_energy:.2e} J/m³ ({elastic_energy/1e6:.2f} MJ/m³)")
print(f"   Plastic energy: {plastic_energy:.2e} J/m³ ({plastic_energy/1e6:.2f} MJ/m³)")
print(f"   (Energy = area under stress-strain curve)")

# 6. Ductility Analysis
classification, description, elongation = analyze_ductility(strain[-1])
print(f"\n6. MATERIAL CLASSIFICATION")
print(f"   Type: {classification}")
print(f"   Elongation: {elongation:.2f}%")
print(f"   Description: {description}")

print(f"\n7. COMPARISON WITH COMMON MATERIALS (A-Level Examples)")
materials_E = {
    'Steel': 200,
    'Copper': 120,
    'Aluminium': 70,
    'Glass': 70,
    'Rubber': 0.01,
    'Wood': 11
}
print(f"   Your material E: {E/1e9:.1f} GPa")
print(f"\n   Common materials:")
for material, value in materials_E.items():
    print(f"   • {material}: {value} GPa")

closest = min(materials_E.items(), key=lambda x: abs(x[1] - E/1e9))
print(f"\n   Your material is closest to: {closest[0]}")

visualize_stress_strain(strain, stress, E,
                       (elastic_strain, elastic_stress),
                       (yield_strain, yield_stress),
                       (uts_strain, uts_stress))