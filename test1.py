import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001

t = np.linspace(T_MIN, T_MAX, N)
print(t)
sin_sig = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)

print(sin_sig)

plt.plot(t, sin_sig, 'o')
plt.show()