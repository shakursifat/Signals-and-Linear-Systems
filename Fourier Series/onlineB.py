import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. OOP Framework Definition
# ==========================================
class SignalGenerator:
    """
    Object-oriented framework for generating continuous-time signals.
    """
    def __init__(self, t):
        self.t = t

    def gaussian(self, a, t0=0):
        """
        Generates a Gaussian signal x(t) = e^(-a*(t - t0)^2).
        The t0 parameter handles the OOP time-shifting requirement natively.
        """
        return np.exp(-a * (self.t - t0)**2)

class CFTAnalyzer:
    """
    Object-oriented framework for computing the Continuous Fourier Transform.
    """
    def __init__(self, t, f):
        self.t = t
        self.f = f

    def compute_cft(self, signal):
        """
        Computes the CFT using numerical integration via the trapezoidal rule.
        Strictly avoids np.fft.
        """
        X = np.zeros(len(self.f), dtype=complex)
        for i, freq in enumerate(self.f):
            # Integrand: signal(t) * e^(-j * 2 * pi * f * t)
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapezoid(integrand, self.t)
        return X

# ==========================================
# 2. Signal Construction & Shifting
# ==========================================
# Define time axis t in [-5, 5] with 2000 samples
t = np.linspace(-5, 5, 2000)

# Instantiate the generator
generator = SignalGenerator(t)

# Generate the original signal x(t) with a=1
x = generator.gaussian(a=1, t0=0)

# Construct the shifted signal y(t) = x(t-1) using the OOP framework
t0_shift = 1
y = generator.gaussian(a=1, t0=t0_shift)

# ==========================================
# 3. Continuous Fourier Transform
# ==========================================
# Define frequency axis f in [-10, 10] with 1000 samples
f = np.linspace(-10, 10, 1000)

# Instantiate the analyzer
analyzer = CFTAnalyzer(t, f)

# Compute CFTs
X_f = analyzer.compute_cft(x)
Y_f = analyzer.compute_cft(y)

# ==========================================
# 4. Numerical Verification & Plotting
# ==========================================
plt.figure(figsize=(14, 6))

# Plot Magnitude Spectra
plt.subplot(1, 2, 1)
plt.plot(f, np.abs(X_f), label='|X(f)| (Original)', color='blue', linewidth=2)
plt.plot(f, np.abs(Y_f), label='|Y(f)| (Shifted)', color='red', linestyle='--', linewidth=2)
plt.title('Magnitude Spectra Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Plot Phase Spectra
# Apply a threshold to ignore phase calculation where magnitude is essentially zero
threshold = 1e-3
phase_X = np.where(np.abs(X_f) > threshold, np.angle(X_f), 0)
phase_Y = np.where(np.abs(Y_f) > threshold, np.angle(Y_f), 0)

plt.subplot(1, 2, 2)
plt.plot(f, phase_X, label='∠X(f) (Original)', color='blue', linewidth=2)
plt.plot(f, phase_Y, label='∠Y(f) (Shifted)', color='red', linestyle='--', linewidth=2)
plt.title('Phase Spectra Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 5. Error Analysis
# ==========================================
# (a) Mean Squared Error (MSE) of Magnitude
MSE_mag = np.mean((np.abs(X_f) - np.abs(Y_f))**2)

# (b) Phase Difference Error
# Theoretical predicted phase for Y(f) is ∠X(f) - 2*pi*f*t0
predicted_phase_Y = np.angle(X_f) - 2 * np.pi * f * t0_shift

# Unwrap phases to ensure continuous comparison across the pi / -pi boundary
unwrapped_phase_Y_measured = np.unwrap(np.angle(Y_f))
unwrapped_phase_Y_predicted = np.unwrap(predicted_phase_Y)

# Mask the phase MSE computation to only include meaningful frequencies (where magnitude > threshold)
valid_indices = np.abs(X_f) > threshold
MSE_phase = np.mean((unwrapped_phase_Y_measured[valid_indices] - unwrapped_phase_Y_predicted[valid_indices])**2)

print("--- Error Analysis ---")
print(f"Magnitude MSE: {MSE_mag:.10f}")
print(f"Phase Difference MSE: {MSE_phase:.10f}")