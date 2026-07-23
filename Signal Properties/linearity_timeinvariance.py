import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Setup and System Definitions
# =====================================================================
n = np.arange(-10, 11) # Time index n from -10 to 10

def x_func(n_val):
    """A basic asymmetrical input signal used for testing shifts."""
    return np.where((n_val >= 0) & (n_val <= 4), n_val, 0)

# Define three different systems:
def sys_LTI(x):
    """Linear and Time-Invariant: y[n] = 2 * x[n]"""
    return 2 * x

def sys_nonlinear(x):
    """Non-linear (but Time-Invariant): y[n] = (x[n])^2"""
    return x ** 2

def sys_time_variant(n_val, x):
    """Time-Variant (but Linear): y[n] = n * x[n]"""
    return n_val * x

# Create two distinct signals for testing Linearity (pulses are easier to visually compare than noise)
x1 = np.where((n >= -2) & (n <= 2), 1.0, 0.0)  # Pulse from -2 to 2
x2 = np.where((n >= 0) & (n <= 4), 0.5, 0.0)   # Pulse from 0 to 4
a, b = 2.0, 3.0 # Constants for scaling

# =====================================================================
# 2. Testing & Plotting Linearity
# =====================================================================
print("--- Linearity Test ---")

# --- LTI System ---
y_combined_input_lti = sys_LTI(a * x1 + b * x2)
y_combined_output_lti = a * sys_LTI(x1) + b * sys_LTI(x2)
is_lti_linear = np.allclose(y_combined_input_lti, y_combined_output_lti)
print(f"Is sys_LTI linear? {is_lti_linear}")

# --- Nonlinear System ---
y_combined_input_nl = sys_nonlinear(a * x1 + b * x2)
y_combined_output_nl = a * sys_nonlinear(x1) + b * sys_nonlinear(x2)
is_nl_linear = np.allclose(y_combined_input_nl, y_combined_output_nl)
print(f"Is sys_nonlinear linear? {is_nl_linear}")

# --- Linearity Figure ---
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))
fig1.suptitle('Testing Linearity: System(a*x1 + b*x2)  vs  a*System(x1) + b*System(x2)', fontsize=14)

# Plot LTI System Results (Should Match Exactly)
axs1[0, 0].stem(n, y_combined_input_lti, basefmt="black")
axs1[0, 0].set_title('LTI System: Output of Combined Input')
axs1[0, 1].stem(n, y_combined_output_lti, basefmt="black", linefmt='C1-', markerfmt='C1o')
axs1[0, 1].set_title('LTI System: Combined Output of Individual Inputs')

# Plot Nonlinear System Results (Will NOT Match)
axs1[1, 0].stem(n, y_combined_input_nl, basefmt="black")
axs1[1, 0].set_title('Nonlinear System: Output of Combined Input')
axs1[1, 1].stem(n, y_combined_output_nl, basefmt="black", linefmt='C3-', markerfmt='C3o')
axs1[1, 1].set_title('Nonlinear System: Combined Output of Individual Inputs')

for ax in axs1.flat:
    ax.grid(True)
plt.tight_layout()


# =====================================================================
# 3. Testing & Plotting Time-Invariance
# =====================================================================
print("\n--- Time-Invariance Test ---")

shift_amount = 3
x_shifted = x_func(n - shift_amount)

# --- LTI System ---
output_of_shifted_lti = sys_LTI(x_shifted)
shifted_output_lti = sys_LTI(x_func(n - shift_amount)) 
is_lti_ti = np.allclose(output_of_shifted_lti, shifted_output_lti)
print(f"Is sys_LTI time-invariant? {is_lti_ti}")

# --- Time-Variant System ---
# 1. Output when given a shifted input: System(x[n-k])
output_of_shifted_tv = sys_time_variant(n, x_shifted)
# 2. Shifted version of the normal output: y[n-k] = (n-k) * x[n-k]
shifted_output_tv = sys_time_variant(n - shift_amount, x_shifted)

is_tv_ti = np.allclose(output_of_shifted_tv, shifted_output_tv)
print(f"Is sys_time_variant time-invariant? {is_tv_ti}")

# --- Time-Invariance Figure ---
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Testing Time-Invariance: System(Shifted Input)  vs  Shifted Output', fontsize=14)

# Plot LTI System Results (Should Match Exactly)
axs2[0, 0].stem(n, output_of_shifted_lti, basefmt="black")
axs2[0, 0].set_title('LTI System: Output when given Shifted Input')
axs2[0, 1].stem(n, shifted_output_lti, basefmt="black", linefmt='C1-', markerfmt='C1o')
axs2[0, 1].set_title('LTI System: Shifted Output of Original Input')

# Plot Time-Variant System Results (Will NOT Match)
axs2[1, 0].stem(n, output_of_shifted_tv, basefmt="black")
axs2[1, 0].set_title('Time-Variant System: Output when given Shifted Input')
axs2[1, 1].stem(n, shifted_output_tv, basefmt="black", linefmt='C3-', markerfmt='C3o')
axs2[1, 1].set_title('Time-Variant System: Shifted Output of Original Input')

for ax in axs2.flat:
    ax.grid(True)
plt.tight_layout()

# Show both figures
plt.show()