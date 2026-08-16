import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Initialization and Signal Definitions
# ==========================================
# Define time array (finite window for numerical integration)
t = np.linspace(-20, 20, 4000)

# Define frequency array (Hz)
# The angular frequencies are w1 = 4 rad/s (f ~= 0.636 Hz) and w2 = 6 rad/s (f ~= 0.955 Hz)
f = np.linspace(-1.5, 1.5, 1000)

# Define the base signal and its analytical derivatives
x = 0.5 * np.cos(4 * t) + 0.5 * np.sin(6 * t)
y1 = -2.0 * np.sin(4 * t) + 3.0 * np.cos(6 * t)
y2 = -8.0 * np.cos(4 * t) - 18.0 * np.sin(6 * t)
y3 = 32.0 * np.sin(4 * t) - 108.0 * np.cos(6 * t)

# ==========================================
# 2. Continuous Fourier Transform Function
# ==========================================
def compute_cft(signal, t_array, f_array):
    """
    Computes the CFT using numerical integration via trapezoidal rule.
    """
    X = np.zeros(len(f_array), dtype=complex)
    for i, freq in enumerate(f_array):
        # Integrand: x(t) * e^(-j * 2 * pi * f * t)
        integrand = signal * np.exp(-1j * 2 * np.pi * freq * t_array)
        X[i] = np.trapezoid(integrand, t_array)
    return X

# Compute base CFT for X(f)
X_f = compute_cft(x, t, f)

# ==========================================
# 3. Property-Based vs Numerical CFTs
# ==========================================
# Calculate property-based CFTs: (j2pi*f)^n * X(f)
j2pif = 1j * 2 * np.pi * f
Y1_prop = j2pif * X_f
Y2_prop = (j2pif**2) * X_f
Y3_prop = (j2pif**3) * X_f

# Calculate numerical CFTs from the derivative time-domain signals
Y1_num = compute_cft(y1, t, f)
Y2_num = compute_cft(y2, t, f)
Y3_num = compute_cft(y3, t, f)

# ==========================================
# 4. Plotting and Overlap Verification
# ==========================================
def plot_comparison(f, Y_num, Y_prop, derivative_name, figure_num):
    plt.figure(figsize=(12, 6))
    
    # Magnitude Subplot
    plt.subplot(1, 2, 1)
    plt.plot(f, np.abs(Y_num), label='Numerical Integration CFT', color='blue', linewidth=2)
    plt.plot(f, np.abs(Y_prop), label='Theoretical Property CFT', color='red', linestyle='--', linewidth=2)
    plt.title(f'Magnitude Comparison: {derivative_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Phase Subplot
    plt.subplot(1, 2, 2)
    # Thresholding phase noise where magnitude is practically zero
    threshold = 1e-2
    phase_num = np.where(np.abs(Y_num) > threshold, np.angle(Y_num), 0)
    phase_prop = np.where(np.abs(Y_prop) > threshold, np.angle(Y_prop), 0)
    
    plt.plot(f, phase_num, label='Numerical Integration CFT Phase', color='blue', linewidth=2)
    plt.plot(f, phase_prop, label='Theoretical Property CFT Phase', color='red', linestyle='--', linewidth=2)
    plt.title(f'Phase Comparison: {derivative_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot all three derivatives
plot_comparison(f, Y1_num, Y1_prop, "1st Derivative", 1)
plot_comparison(f, Y2_num, Y2_prop, "2nd Derivative", 2)
plot_comparison(f, Y3_num, Y3_prop, "3rd Derivative", 3)

# ==========================================
# 5. MSE Analysis
# ==========================================
def calculate_mse(Y_num, Y_prop):
    mag_mse = np.mean((np.abs(Y_num) - np.abs(Y_prop))**2)
    
    # Use thresholded phase to avoid noise floor blowing up MSE
    threshold = 1e-2
    phase_num = np.where(np.abs(Y_num) > threshold, np.angle(Y_num), 0)
    phase_prop = np.where(np.abs(Y_prop) > threshold, np.angle(Y_prop), 0)
    phase_mse = np.mean((phase_num - phase_prop)**2)
    
    return mag_mse, phase_mse

mse1_mag, mse1_phase = calculate_mse(Y1_num, Y1_prop)
mse2_mag, mse2_phase = calculate_mse(Y2_num, Y2_prop)
mse3_mag, mse3_phase = calculate_mse(Y3_num, Y3_prop)

print("--- Mean Squared Error (MSE) Analysis ---")
print(f"1st Derivative -> Magnitude MSE: {mse1_mag:.6f}, Phase MSE: {mse1_phase:.6f}")
print(f"2nd Derivative -> Magnitude MSE: {mse2_mag:.6f}, Phase MSE: {mse2_phase:.6f}")
print(f"3rd Derivative -> Magnitude MSE: {mse3_mag:.6f}, Phase MSE: {mse3_phase:.6f}")