import numpy as np
import matplotlib.pyplot as plt

# 1. Create an ASYMMETRIC time array (e.g., from -2 to 5)
t = np.linspace(-2, 5, 400)

# 2. Define a base signal x(t). 
# Let's use an exponential decay that only exists for t >= 0
x = np.where(t >= 0, np.exp(-t), 0)


# ==========================================================
# METHOD 1: Interpolation (Best for keeping the same 't' array)
# ==========================================================
# We want to evaluate the signal at -t.
# np.interp requires the reference time axis to be strictly increasing.
# -t is decreasing, so we reverse both arrays to make time increasing again.
t_inverted_increasing = -t[::-1] 
x_reversed = x[::-1]

# Interpolate to find what the reversed signal values are at the ORIGINAL 't' points.
# left=0, right=0 ensures that out-of-bounds values are clamped to 0.
y_interpolated = np.interp(t, t_inverted_increasing, x_reversed, left=0, right=0)


# ==========================================================
# METHOD 2: New Time Axis (Best if you just need to plot it)
# ==========================================================
# Just negate the time axis and reverse both to keep time strictly increasing.
t_new = -t[::-1]
y_new = x[::-1]


# ==========================================================
# Plotting the Results
# ==========================================================
plt.figure(figsize=(10, 6))

# Original Signal
plt.plot(t, x, label='Original $x(t)$', linewidth=3, color='blue')

# Plotting Method 1
plt.plot(t, y_interpolated, label='Method 1: $x(-t)$ (Interpolated on original $t$)', 
         linestyle='--', linewidth=2, color='orange')

# Plotting Method 2 (Shifted slightly as a scatter to prove it matches)
plt.plot(t_new, y_new, label='Method 2: $x(-t)$ (On new $t_{new}$ axis)', 
         linestyle=':', linewidth=3, color='red')

plt.axvline(0, color='black', linewidth=1, linestyle='-') # Y-axis
plt.axhline(0, color='black', linewidth=1, linestyle='-') # X-axis
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Time Reversal of an Asymmetric Signal')
plt.legend()
plt.grid(True)
plt.show()