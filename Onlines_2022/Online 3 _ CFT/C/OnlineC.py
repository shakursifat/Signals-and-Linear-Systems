import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. OOP Framework Definitions
# ==========================================
class SignalGenerator:
    """
    Object-oriented framework for generating continuous-time signals.
    """
    def __init__(self, t):
        self.t = t

    def square(self, t_val):
        """Generates a standard square wave (rect function)."""
        return np.where((t_val >= -0.5) & (t_val <= 0.5), 1.0, 0.0)

    def triangle(self, t_val):
        """Generates a standard triangle wave."""
        return np.where((t_val >= -1.0) & (t_val <= 1.0), 1.0 - np.abs(t_val), 0.0)

    def generate_x(self):
        """Generates the base signal x(t) = Square(t) + Triangle(t)."""
        return self.square(self.t) + self.triangle(self.t)

    def generate_y(self, a, f0):
        """
        Generates the modified signal y(t).
        Applies time compression (a) and phase shift (f0).
        """
        t_scaled = a * self.t
        x_scaled = self.square(t_scaled) + self.triangle(t_scaled)
        phase_shift = np.exp(1j * 2 * np.pi * f0 * self.t)
        return x_scaled * phase_shift

class CFTAnalyzer:
    """
    Object-oriented framework for computing the Continuous Fourier Transform.
    """
    def __init__(self, t):
        self.t = t

    def compute_cft(self, signal, freq_array):
        """
        Computes the CFT using numerical integration via the trapezoidal rule.
        Strictly avoids np.fft.
        """
        X = np.zeros(len(freq_array), dtype=complex)
        for i, freq in enumerate(freq_array):
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapezoid(integrand, self.t)
        return X

class ErrorAnalyzer:
    """
    Object-oriented framework for evaluating Mean Squared Error (MSE).
    """
    @staticmethod
    def calculate_metrics(Y_meas, X_mapped_theo, a):
        """
        Calculates the magnitude and phase MSE based on the provided theoretical formulas.
        """
        # Formulate the theoretical Y(f) for comparison
        Y_theo_mag = (1 / np.abs(a)) * np.abs(X_mapped_theo)
        Y_theo_phase = np.angle(X_mapped_theo)
        
        # (a) Magnitude MSE
        mse_mag = np.mean((np.abs(Y_meas) - Y_theo_mag)**2)
        
        # (b) Phase MSE
        # Use a threshold to ignore artificial phase noise where the magnitude is near zero
        threshold = 1e-3
        valid_indices = Y_theo_mag > threshold
        
        # Unwrap phases to prevent 2*pi boundary jumps from artificially inflating the MSE
        phase_meas_unwrap = np.unwrap(np.angle(Y_meas))
        phase_theo_unwrap = np.unwrap(Y_theo_phase)
        
        mse_phase = np.mean((phase_meas_unwrap[valid_indices] - phase_theo_unwrap[valid_indices])**2)
        
        return mse_mag, mse_phase


# ==========================================
# 2. Initialization and Generation
# ==========================================
# Define time axis t in [-5, 5] using 2000 samples
t = np.linspace(-5, 5, 2000)

# Define frequency axis f in [-10, 10] using 1000 samples
f = np.linspace(-10, 10, 1000)

a = 10
f0 = 10

# Generate signals
generator = SignalGenerator(t)
x = generator.generate_x()
y = generator.generate_y(a, f0)

# ==========================================
# 3. Continuous Fourier Transform
# ==========================================
analyzer = CFTAnalyzer(t)

# Find the standard CFT of x(t) and y(t)
X_f = analyzer.compute_cft(x, f)
Y_f = analyzer.compute_cft(y, f)

# To verify the property analytically, we must compute X at the mapped frequencies: (f - f0) / a
f_mapped = (f - f0) / a
X_mapped = analyzer.compute_cft(x, f_mapped)

# Construct expected theoretical arrays for plotting
Y_expected_mag = (1 / np.abs(a)) * np.abs(X_mapped)
Y_expected_phase = np.angle(X_mapped)

# ==========================================
# 4. Numerical Verification & Plotting
# ==========================================
plt.figure(figsize=(14, 6))

# Plot Magnitude Spectra
plt.subplot(1, 2, 1)
plt.plot(f, np.abs(Y_f), label='|Y(f)| (Measured)', color='blue', linewidth=2)
plt.plot(f, Y_expected_mag, label='1/|a| * |X((f-f0)/a)| (Theoretical)', color='red', linestyle='--', linewidth=2)
plt.title('Magnitude Spectra Verification')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Plot Phase Spectra (Thresholded for clean visual output)
threshold_val = 1e-3
phase_Y_plot = np.where(np.abs(Y_f) > threshold_val, np.angle(Y_f), 0)
phase_X_mapped_plot = np.where(Y_expected_mag > threshold_val, Y_expected_phase, 0)

plt.subplot(1, 2, 2)
plt.plot(f, phase_Y_plot, label='∠Y(f) (Measured)', color='blue', linewidth=2)
plt.plot(f, phase_X_mapped_plot, label='∠X((f-f0)/a) (Theoretical)', color='red', linestyle='--', linewidth=2)
plt.title('Phase Spectra Verification')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 5. Error Analysis
# ==========================================
error_analyzer = ErrorAnalyzer()
mse_mag, mse_phase = error_analyzer.calculate_metrics(Y_f, X_mapped, a)

print("--- Error Analysis ---")
print(f"MSE Magnitude: {mse_mag:.10f}")
print(f"MSE Phase: {mse_phase:.10f}")
if mse_mag < 1e-3 and mse_phase < 1e-3:
    print("Verification Successful: Both MSE values fall within the acceptable tolerance range.")