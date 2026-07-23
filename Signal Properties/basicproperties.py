import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup Base Time Vector and Signal
# ---------------------------------------------------------
n = np.arange(-10, 11) # Time index n from -10 to 10

# Define a simple non-symmetric signal: x[n] = n for 0 <= n <= 4, else 0
def x_func(n_val):
    # Using np.where to apply conditions element-wise
    return np.where((n_val >= 0) & (n_val <= 4), n_val, 0)

x = x_func(n)

# ---------------------------------------------------------
# 2. Operations
# ---------------------------------------------------------
# Time Delay (Shift Right by 3): x[n - 3]
x_delayed = x_func(n - 3) 

# Time Advance (Shift Left by 3): x[n + 3]
x_advanced = x_func(n + 3)

# Time Reversal (Folding): x[-n]
x_reversed = x_func(-n)

# Causality Check: A signal is causal if x[n] == 0 for all n < 0
is_causal = np.all(x[n < 0] == 0)
print(f"Is the base signal causal? {is_causal}")

# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.5)

# Plot Original Signal
axs[0].stem(n, x, basefmt="black")
axs[0].set_title('Original Signal $x[n]$')
axs[0].grid(True)

# Plot Delayed Signal
axs[1].stem(n, x_delayed, basefmt="black", linefmt='C1-', markerfmt='C1o')
axs[1].set_title('Time Delayed Signal $x[n-3]$')
axs[1].grid(True)

# Plot Advanced Signal
axs[2].stem(n, x_advanced, basefmt="black", linefmt='C2-', markerfmt='C2o')
axs[2].set_title('Time Advanced Signal $x[n+3]$')
axs[2].grid(True)

# Plot Reversed Signal
axs[3].stem(n, x_reversed, basefmt="black", linefmt='C3-', markerfmt='C3o')
axs[3].set_title('Time Reversed Signal $x[-n]$')
axs[3].grid(True)

plt.show()