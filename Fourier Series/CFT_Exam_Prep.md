# CSE 220 — Continuous Fourier Transform (CFT) Exam Prep Guide

> Based on: Online A1/A2, B1/B2, C1/C2 patterns + Offline spec (Task 1 & Task 2)
> **RULE: No `np.fft` allowed anywhere. Use `np.trapezoid` (or `np.trapz`) for ALL integrals.**

---

## Table of Contents

1. [Core Formulas](#1-core-formulas)
2. [CFT Properties Summary](#2-cft-properties-summary)
3. [Universal CFT OOP Template](#3-universal-cft-oop-template)
4. [Problem Type A — Differentiation Property](#4-problem-type-a--differentiation-property)
5. [Problem Type B — Time-Shift Property](#5-problem-type-b--time-shift-property)
6. [Problem Type C — Time-Scaling + Modulation](#6-problem-type-c--time-scaling--modulation)
7. [Practice Variations](#7-practice-variations)
8. [Task 1 — Fourier Series Epicycles](#8-task-1--fourier-series-epicycles)
9. [Task 2 — 2D CFT Edge Detector](#9-task-2--2d-cft-edge-detector)
10. [Common Mistakes & Exam Checklist](#10-common-mistakes--exam-checklist)

---

## 1. Core Formulas

### 1D CFT Forward
```
X(f) = integral[-inf, +inf]  x(t) * e^(-j*2*pi*f*t)  dt
```
Numerical:
```python
X[i] = np.trapezoid(signal * np.exp(-1j * 2 * np.pi * f[i] * t), t)
```

### 1D CFT Inverse
```
x(t) = integral[-inf, +inf]  X(f) * e^(j*2*pi*f*t)  df
```

### 2D CFT Forward (separable)
```
Re{F(u,v)} =  integral integral  I(x,y)*cos(2*pi*(u*x + v*y))  dx dy
Im{F(u,v)} = -integral integral  I(x,y)*sin(2*pi*(u*x + v*y))  dx dy
```

### 2D Inverse CFT
```
I(x,y) = integral integral  F(u,v)*e^(j*2*pi*(u*x + v*y))  du dv
```

---

## 2. CFT Properties Summary

| Property | Time Domain | Frequency Domain |
|---|---|---|
| Linearity | `a*x(t) + b*y(t)` | `a*X(f) + b*Y(f)` |
| **Time Shift** | `x(t - t0)` | `X(f) * e^(-j*2*pi*f*t0)` |
| **Differentiation** | `d^n x/dt^n` | `(j*2*pi*f)^n * X(f)` |
| **Time Scaling** | `x(a*t)` | `(1/|a|) * X(f/a)` |
| **Modulation** | `x(t) * e^(j*2*pi*f0*t)` | `X(f - f0)` |
| **Combined** | `x(a*t) * e^(j*2*pi*f0*t)` | `(1/|a|) * X((f-f0)/a)` |
| Mag (time-shift) | unchanged | `|Y(f)| = |X(f)|` |
| Phase (time-shift) | linear phase added | `ang(Y) = ang(X) - 2*pi*f*t0` |

---

## 3. Universal CFT OOP Template

Reusable for every exam question. Adapt signal definition and property.

```python
import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, t):
        self.t = t

    def cosine(self, A, omega):
        return A * np.cos(omega * self.t)

    def sine(self, A, omega):
        return A * np.sin(omega * self.t)

    def gaussian(self, a, t0=0):
        # e^(-a*(t-t0)^2)
        return np.exp(-a * (self.t - t0)**2)

    def rect(self, t_val=None):
        # rect pulse: 1 for |t| <= 0.5
        tv = t_val if t_val is not None else self.t
        return np.where((tv >= -0.5) & (tv <= 0.5), 1.0, 0.0)

    def triangle(self, t_val=None):
        # triangle: 1-|t| for |t| <= 1
        tv = t_val if t_val is not None else self.t
        return np.where((tv >= -1.0) & (tv <= 1.0), 1.0 - np.abs(tv), 0.0)


class CFTAnalyzer:
    def __init__(self, t, f=None):
        self.t = t
        self.f = f

    def compute_cft(self, signal, freq_array=None):
        # X(f) = integral signal * e^(-j*2*pi*f*t) dt
        # freq_array overrides self.f if provided
        f_use = freq_array if freq_array is not None else self.f
        X = np.zeros(len(f_use), dtype=complex)
        for i, freq in enumerate(f_use):
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapezoid(integrand, self.t)
        return X


class ErrorAnalyzer:
    THRESHOLD = 1e-2

    def magnitude_mse(self, Y_meas, Y_theory):
        return np.mean((np.abs(Y_meas) - np.abs(Y_theory))**2)

    def phase_mse(self, Y_meas, Y_theory, threshold=None):
        thr = threshold or self.THRESHOLD
        valid = np.abs(Y_theory) > thr
        pm = np.unwrap(np.angle(Y_meas))
        pt = np.unwrap(np.angle(Y_theory))
        return np.mean((pm[valid] - pt[valid])**2)

    def report(self, label, Y_meas, Y_theory):
        m = self.magnitude_mse(Y_meas, Y_theory)
        p = self.phase_mse(Y_meas, Y_theory)
        print(f"[{label}] Magnitude MSE: {m:.8f}  |  Phase MSE: {p:.8f}")
        if m < 1e-3 and p < 1e-3:
            print("  -> VERIFIED: both MSE values within tolerance")
        return m, p


def plot_comparison(f, Y_num, Y_prop, title, threshold=1e-2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(f, np.abs(Y_num),  'b-',  lw=2, label='Numerical CFT')
    axes[0].plot(f, np.abs(Y_prop), 'r--', lw=2, label='Theoretical (Property)')
    axes[0].set_title(f'Magnitude: {title}')
    axes[0].set_xlabel('Frequency (Hz)'); axes[0].set_ylabel('|X(f)|')
    axes[0].legend(); axes[0].grid(True)
    pn = np.where(np.abs(Y_num)  > threshold, np.angle(Y_num),  0)
    pp = np.where(np.abs(Y_prop) > threshold, np.angle(Y_prop), 0)
    axes[1].plot(f, pn, 'b-',  lw=2, label='Numerical Phase')
    axes[1].plot(f, pp, 'r--', lw=2, label='Theoretical Phase')
    axes[1].set_title(f'Phase: {title}')
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('Phase (rad)')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.show()
```

---

## 4. Problem Type A — Differentiation Property

### Theory
```
d^n x(t)/dt^n  --[CFT]-->  (j*2*pi*f)^n * X(f)
```

### Sample Problem (Online A1/A2 style)
**Given:** `x(t) = 0.5*cos(4t) + 0.5*sin(6t)`
**Task:** Verify differentiation property for 1st, 2nd, 3rd derivatives. Plot and MSE.

**Derivatives (compute by hand):**
```
y1 = dx/dt   = -2*sin(4t) + 3*cos(6t)
y2 = d2x/dt2 = -8*cos(4t) - 18*sin(6t)
y3 = d3x/dt3 = 32*sin(4t) - 108*cos(6t)
```

**Derivative rules:**
```
d/dt [A*cos(w*t)] = -A*w*sin(w*t)
d/dt [A*sin(w*t)] = +A*w*cos(w*t)
```

### Complete Working Code

```python
import numpy as np
import matplotlib.pyplot as plt

# CHANGE: signal x(t), derivatives y1/y2/y3, f range
t = np.linspace(-20, 20, 4000)    # wide window for sinusoids
f = np.linspace(-1.5, 1.5, 1000)  # adjust to signal bandwidth

# --- DEFINE SIGNAL (change per exam) ---
# w1=4 rad/s -> f1~0.64 Hz, w2=6 rad/s -> f2~0.95 Hz
x  = 0.5 * np.cos(4 * t) + 0.5 * np.sin(6 * t)
y1 = -2.0 * np.sin(4 * t) + 3.0 * np.cos(6 * t)      # 1st derivative
y2 = -8.0 * np.cos(4 * t) - 18.0 * np.sin(6 * t)     # 2nd derivative
y3 = 32.0 * np.sin(4 * t) - 108.0 * np.cos(6 * t)    # 3rd derivative

def compute_cft(signal, t_arr, f_arr):
    X = np.zeros(len(f_arr), dtype=complex)
    for i, freq in enumerate(f_arr):
        integrand = signal * np.exp(-1j * 2 * np.pi * freq * t_arr)
        X[i] = np.trapezoid(integrand, t_arr)
    return X

X_f    = compute_cft(x, t, f)
Y1_num = compute_cft(y1, t, f)
Y2_num = compute_cft(y2, t, f)
Y3_num = compute_cft(y3, t, f)

# Property-based: (j*2*pi*f)^n * X(f)
j2pif   = 1j * 2 * np.pi * f
Y1_prop = j2pif    * X_f
Y2_prop = j2pif**2 * X_f
Y3_prop = j2pif**3 * X_f

def plot_cmp(f, Y_num, Y_prop, label, thr=1e-2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(f, np.abs(Y_num),  'b-',  lw=2, label='Numerical CFT')
    ax1.plot(f, np.abs(Y_prop), 'r--', lw=2, label='Property CFT')
    ax1.set_title(f'Magnitude: {label}'); ax1.set_xlabel('f (Hz)'); ax1.legend(); ax1.grid(True)
    pn = np.where(np.abs(Y_num)  > thr, np.angle(Y_num),  0)
    pp = np.where(np.abs(Y_prop) > thr, np.angle(Y_prop), 0)
    ax2.plot(f, pn, 'b-',  lw=2, label='Numerical Phase')
    ax2.plot(f, pp, 'r--', lw=2, label='Property Phase')
    ax2.set_title(f'Phase: {label}'); ax2.set_xlabel('f (Hz)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()

plot_cmp(f, Y1_num, Y1_prop, '1st Derivative')
plot_cmp(f, Y2_num, Y2_prop, '2nd Derivative')
plot_cmp(f, Y3_num, Y3_prop, '3rd Derivative')

def mse(Y_num, Y_prop, thr=1e-2):
    mag = np.mean((np.abs(Y_num) - np.abs(Y_prop))**2)
    pn  = np.where(np.abs(Y_num)  > thr, np.angle(Y_num),  0)
    pp  = np.where(np.abs(Y_prop) > thr, np.angle(Y_prop), 0)
    return mag, np.mean((pn - pp)**2)

m1, p1 = mse(Y1_num, Y1_prop)
m2, p2 = mse(Y2_num, Y2_prop)
m3, p3 = mse(Y3_num, Y3_prop)

print("--- Differentiation Property MSE ---")
print(f"1st Deriv -> Mag MSE: {m1:.8f}, Phase MSE: {p1:.8f}")
print(f"2nd Deriv -> Mag MSE: {m2:.8f}, Phase MSE: {p2:.8f}")
print(f"3rd Deriv -> Mag MSE: {m3:.8f}, Phase MSE: {p3:.8f}")
```

**Adaptation for different signal** (e.g. `x(t) = 3*cos(2t) + sin(5t)`):
```python
x  = 3 * np.cos(2*t) + np.sin(5*t)
y1 = -6 * np.sin(2*t) + 5  * np.cos(5*t)   # -A*w*sin + A*w*cos
y2 = -12 * np.cos(2*t) - 25 * np.sin(5*t)  # -A*w^2*cos - A*w^2*sin
y3 = 24 * np.sin(2*t) - 125 * np.cos(5*t)  # A*w^3*sin - A*w^3*cos
```

---

## 5. Problem Type B — Time-Shift Property

### Theory
```
y(t) = x(t - t0)  --[CFT]-->  Y(f) = X(f) * e^(-j*2*pi*f*t0)

Key results:
  |Y(f)| = |X(f)|                           -> Magnitude MSE ~ 0
  angle(Y) = angle(X) - 2*pi*f*t0           -> Phase shifted linearly
```

### Sample Problem (Online B1/B2 style)
**Given:** `x(t) = e^(-t^2)` (Gaussian, a=1), `t0 = 1`
**Task:** Construct `y(t) = x(t-1)` using OOP. Compute CFT, verify, compute MSE.

### Complete Working Code

```python
import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, t):
        self.t = t

    def gaussian(self, a, t0=0):
        # Built-in time-shift via t0 parameter
        return np.exp(-a * (self.t - t0)**2)

    def rect(self, t0=0):
        tv = self.t - t0
        return np.where((tv >= -0.5) & (tv <= 0.5), 1.0, 0.0)

    def triangle(self, t0=0):
        tv = self.t - t0
        return np.where((tv >= -1.0) & (tv <= 1.0), 1.0 - np.abs(tv), 0.0)

    def cosine_shifted(self, A, omega, t0=0):
        return A * np.cos(omega * (self.t - t0))


class CFTAnalyzer:
    def __init__(self, t, f):
        self.t = t; self.f = f

    def compute_cft(self, signal):
        X = np.zeros(len(self.f), dtype=complex)
        for i, freq in enumerate(self.f):
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapezoid(integrand, self.t)
        return X

# Setup — CHANGE values per exam question
t  = np.linspace(-5, 5, 2000)
f  = np.linspace(-10, 10, 1000)
t0 = 1   # TIME SHIFT

gen      = SignalGenerator(t)
analyzer = CFTAnalyzer(t, f)

x = gen.gaussian(a=1, t0=0)    # original x(t)
y = gen.gaussian(a=1, t0=t0)   # shifted y(t)=x(t-t0), built via OOP

X_f = analyzer.compute_cft(x)
Y_f = analyzer.compute_cft(y)
Y_prop = X_f * np.exp(-1j * 2 * np.pi * f * t0)  # theoretical

# Plot
threshold = 1e-3
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(f, np.abs(X_f), 'b-',  lw=2, label='|X(f)| Original')
axes[0].plot(f, np.abs(Y_f), 'r--', lw=2, label='|Y(f)| Shifted (Numerical)')
axes[0].set_title('Magnitude Spectra'); axes[0].set_xlabel('f (Hz)')
axes[0].legend(); axes[0].grid(True)

phase_X = np.where(np.abs(X_f) > threshold, np.angle(X_f), 0)
phase_Y = np.where(np.abs(Y_f) > threshold, np.angle(Y_f), 0)
axes[1].plot(f, phase_X, 'b-',  lw=2, label='Phase X(f) Original')
axes[1].plot(f, phase_Y, 'r--', lw=2, label='Phase Y(f) Shifted')
axes[1].set_title('Phase Spectra'); axes[1].set_xlabel('f (Hz)')
axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

# MSE
MSE_mag = np.mean((np.abs(X_f) - np.abs(Y_f))**2)

predicted_phase   = np.angle(X_f) - 2 * np.pi * f * t0
valid             = np.abs(X_f) > threshold
py_unwrap         = np.unwrap(np.angle(Y_f))
pred_unwrap       = np.unwrap(predicted_phase)
MSE_phase = np.mean((py_unwrap[valid] - pred_unwrap[valid])**2)

print("--- Time-Shift Property MSE ---")
print(f"Magnitude MSE: {MSE_mag:.10f}  (should be ~0)")
print(f"Phase MSE:     {MSE_phase:.10f}  (should be ~0)")
```

---

## 6. Problem Type C — Time-Scaling + Modulation

### Theory
```
y(t) = x(a*t) * e^(j*2*pi*f0*t)  --[CFT]-->  Y(f) = (1/|a|) * X((f-f0)/a)

So:
  |Y(f)| = (1/|a|) * |X((f-f0)/a)|
  angle(Y) = angle(X((f-f0)/a))
```

### Sample Problem (Online C1/C2 style)
**Given:** `x(t) = Square(t) + Triangle(t)`, `a = 10`, `f0 = 10`
**Task:** Build `y(t) = x(10t)*e^(j*2*pi*10*t)`, verify property, compute MSE.

### Complete Working Code

```python
import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, t):
        self.t = t

    def rect(self, t_val):
        return np.where((t_val >= -0.5) & (t_val <= 0.5), 1.0, 0.0)

    def triangle(self, t_val):
        return np.where((t_val >= -1.0) & (t_val <= 1.0), 1.0 - np.abs(t_val), 0.0)

    def generate_x(self):
        # Base signal x(t) — CHANGE per question
        return self.rect(self.t) + self.triangle(self.t)

    def generate_y(self, a, f0):
        # y(t) = x(a*t) * e^(j*2*pi*f0*t)
        t_scaled   = a * self.t
        x_comp     = self.rect(t_scaled) + self.triangle(t_scaled)
        modulation = np.exp(1j * 2 * np.pi * f0 * self.t)
        return x_comp * modulation


class CFTAnalyzer:
    def __init__(self, t):
        self.t = t

    def compute_cft(self, signal, freq_array):
        X = np.zeros(len(freq_array), dtype=complex)
        for i, freq in enumerate(freq_array):
            integrand = signal * np.exp(-1j * 2 * np.pi * freq * self.t)
            X[i] = np.trapezoid(integrand, self.t)
        return X


class ErrorAnalyzer:
    @staticmethod
    def calculate_metrics(Y_meas, X_mapped, a, threshold=1e-3):
        Y_theo_mag = (1 / np.abs(a)) * np.abs(X_mapped)
        mse_mag    = np.mean((np.abs(Y_meas) - Y_theo_mag)**2)
        valid      = Y_theo_mag > threshold
        pm         = np.unwrap(np.angle(Y_meas))
        pt         = np.unwrap(np.angle(X_mapped))
        mse_phase  = np.mean((pm[valid] - pt[valid])**2)
        return mse_mag, mse_phase

# Setup — CHANGE per exam question
t  = np.linspace(-5, 5, 2000)
f  = np.linspace(-10, 10, 1000)
a  = 10    # time-compression factor
f0 = 10    # frequency shift

gen = SignalGenerator(t)
x   = gen.generate_x()
y   = gen.generate_y(a, f0)

analyzer = CFTAnalyzer(t)
X_f      = analyzer.compute_cft(x, f)
Y_f      = analyzer.compute_cft(y, f)

f_mapped       = (f - f0) / a
X_mapped       = analyzer.compute_cft(x, f_mapped)
Y_expected_mag = (1 / np.abs(a)) * np.abs(X_mapped)
Y_exp_phase    = np.angle(X_mapped)

# Plot
threshold = 1e-3
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(f, np.abs(Y_f),     'b-',  lw=2, label='|Y(f)| Measured')
axes[0].plot(f, Y_expected_mag,  'r--', lw=2, label='(1/|a|)|X((f-f0)/a)| Theoretical')
axes[0].set_title('Magnitude Verification'); axes[0].set_xlabel('f (Hz)')
axes[0].legend(); axes[0].grid(True)

pY  = np.where(np.abs(Y_f)     > threshold, np.angle(Y_f), 0)
pXm = np.where(Y_expected_mag  > threshold, Y_exp_phase,   0)
axes[1].plot(f, pY,  'b-',  lw=2, label='Phase Y(f) Measured')
axes[1].plot(f, pXm, 'r--', lw=2, label='Phase X((f-f0)/a) Theoretical')
axes[1].set_title('Phase Verification'); axes[1].set_xlabel('f (Hz)')
axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

err = ErrorAnalyzer()
mse_m, mse_p = err.calculate_metrics(Y_f, X_mapped, a)
print("--- Time-Scaling + Modulation MSE ---")
print(f"MSE Magnitude: {mse_m:.10f}")
print(f"MSE Phase:     {mse_p:.10f}")
if mse_m < 1e-3 and mse_p < 1e-3:
    print("Verification Successful: both MSE values within acceptable tolerance.")
```

---

## 7. Practice Variations

### Sinc + time-shift
```python
t  = np.linspace(-10, 10, 4000)
f  = np.linspace(-5, 5, 1000)
t0 = 0.5
x  = np.sinc(2 * t)          # normalized: sin(pi*t)/(pi*t)
y  = np.sinc(2 * (t - t0))   # time-shifted
# Theory: |Y| = |X|,  angle(Y) = angle(X) - 2*pi*f*t0
```

### Gaussian + scaling only
```python
t = np.linspace(-5, 5, 2000)
f = np.linspace(-10, 10, 1000)
a = 2
x = np.exp(-t**2)
y = np.exp(-(a * t)**2)    # = e^(-4t^2)
# Theory: Y(f) = (1/2) * X(f/2)
f_mapped = f / a
# X_mapped = compute_cft(x, f_mapped)
# Y_theo_mag = (1/a) * |X_mapped|
```

### Rect + shift + modulation
```python
t  = np.linspace(-5, 5, 2000)
f  = np.linspace(-10, 10, 1000)
t0 = 1;  f0 = 2
x    = np.where((t >= -0.5) & (t <= 0.5), 1.0, 0.0)
x_sh = np.where(((t-t0) >= -0.5) & ((t-t0) <= 0.5), 1.0, 0.0)
y    = x_sh * np.exp(1j * 2 * np.pi * f0 * t)
# Theory: |Y(f)| = |X(f-f0)|
```

### Derivative Quick Reference

| Signal | 1st derivative | 2nd derivative |
|---|---|---|
| `A*cos(w*t)` | `-A*w*sin(w*t)` | `-A*w^2*cos(w*t)` |
| `A*sin(w*t)` | `+A*w*cos(w*t)` | `-A*w^2*sin(w*t)` |
| `e^(-a*t^2)` | `-2*a*t*e^(-a*t^2)` | `(-2a + 4a^2*t^2)*e^(-a*t^2)` |

---

## 8. Task 1 — Fourier Series Epicycles

**Formulas:**
```
c_n    = (1/T) * integral[0,T]  f(t) * e^(-j*n*omega*t)  dt
f_hat  = sum_{n=-N}^{N}  c_n * e^(j*n*omega*t)
omega  = 2*pi / T
```

```python
import numpy as np

class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        # t: closed [0, T] array (t[0]=0, t[-1]=T)
        # signal: complex 1D array z[i] = x(t[i]) + j*y(t[i])
        # n_harmonics: N (range -N to +N, total 2N+1 terms)
        self.t      = t
        self.signal = signal
        self.N      = n_harmonics
        self.T      = t[-1] - t[0]         # Period (NOT just t[-1])
        self.omega  = 2 * np.pi / self.T   # Fundamental angular freq
        self.coeffs = {}                    # will hold {n: c_n}

    def calculate_cn(self, n):
        # c_n = (1/T) * integral  f(t)*e^(-j*n*omega*t)  dt
        integrand = self.signal * np.exp(-1j * n * self.omega * self.t)
        return (1 / self.T) * np.trapezoid(integrand, self.t)

    def calculate_all_coefficients(self):
        # range(-N, N+1) includes BOTH -N and +N
        for n in range(-self.N, self.N + 1):
            self.coeffs[n] = self.calculate_cn(n)

    def approximate(self, t):
        # f_hat = sum c_n * e^(j*n*omega*t) — supports scalar AND array t
        t_arr = np.asarray(t)
        f_hat = np.zeros_like(t_arr, dtype=complex)
        for n in range(-self.N, self.N + 1):
            f_hat += self.coeffs[n] * np.exp(1j * n * self.omega * t_arr)
        if np.isscalar(t):
            return f_hat.item()
        return f_hat
```

**Key reminders:**
- `self.T = t[-1] - t[0]`  (closed interval, so T = 2*pi, not just t[-1])
- Loop `range(-N, N+1)` — includes -N AND +N
- `approximate()` — handle both scalar and array via `np.asarray(t)`

---

## 9. Task 2 — 2D CFT Edge Detector

**Key identity (separability):**
```
cos(2*pi*(ux + vy)) = cos(ux)*cos(vy) - sin(ux)*sin(vy)
```
Integrate over x first (Stage 1), then y (Stage 2) → O(N^3) not O(N^4).

**CRITICAL: Use `self.u / self.v` as freq axes, NOT `self.x / self.y`.**

```python
import numpy as np

class CFT2D:
    def __init__(self, image_obj):
        self.I = image_obj.image    # shape (Ny, Nx)
        self.x = image_obj.x       # (Nx,)
        self.y = image_obj.y       # (Ny,)
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        # Frequency axes at Nyquist range — USE THESE not x/y
        self.u = np.linspace(-1/(2*dx), 1/(2*dx), self.I.shape[1])
        self.v = np.linspace(-1/(2*dy), 1/(2*dy), self.I.shape[0])

    def compute_cft(self):
        # Returns real_F, imag_F each of shape (Nv, Nu)
        Ny, Nx = self.I.shape
        Nu = len(self.u)
        Nv = len(self.v)

        # Stage 1: for each u, integrate over x
        Ix_c = np.zeros((Ny, Nu))
        Ix_s = np.zeros((Ny, Nu))
        for i in range(Nu):
            cos_ux = np.cos(2 * np.pi * self.u[i] * self.x)
            sin_ux = np.sin(2 * np.pi * self.u[i] * self.x)
            Ix_c[:, i] = np.trapezoid(self.I * cos_ux, self.x, axis=1)
            Ix_s[:, i] = np.trapezoid(self.I * sin_ux, self.x, axis=1)

        # Stage 2: for each v, integrate over y
        real_F = np.zeros((Nv, Nu))
        imag_F = np.zeros((Nv, Nu))
        for j in range(Nv):
            cos_vy = np.cos(2 * np.pi * self.v[j] * self.y)
            sin_vy = np.sin(2 * np.pi * self.v[j] * self.y)
            # cos(ux+vy) = cos(ux)*cos(vy) - sin(ux)*sin(vy)
            rp = (Ix_c * cos_vy[:, np.newaxis]) - (Ix_s * sin_vy[:, np.newaxis])
            # -sin(ux+vy) = -(sin(ux)*cos(vy) + cos(ux)*sin(vy))
            ip = -((Ix_s * cos_vy[:, np.newaxis]) + (Ix_c * sin_vy[:, np.newaxis]))
            real_F[j, :] = np.trapezoid(rp, self.y, axis=0)
            imag_F[j, :] = np.trapezoid(ip, self.y, axis=0)

        return real_F, imag_F

    def plot_magnitude(self):
        real, imag = self.compute_cft()
        magnitude  = np.sqrt(real**2 + imag**2)
        import matplotlib.pyplot as plt
        plt.imshow(np.log(1 + magnitude), cmap='magma')
        plt.title('2D CFT Magnitude Spectrum (log-scaled)')
        plt.axis('off'); plt.show()


class InverseCFT2D:
    def __init__(self, real, imag, u, v, x, y):
        self.real = real; self.imag = imag
        self.u = u; self.v = v; self.x = x; self.y = y

    def reconstruct(self):
        # I(x,y) = integral integral F(u,v)*e^(j*2*pi*(ux+vy)) du dv
        # Returns shape (Ny, Nx)
        Ny = len(self.y); Nx = len(self.x); Nu = len(self.u)

        # Stage 1: for each y, integrate F over v
        F_c = np.zeros((Ny, Nu))
        F_s = np.zeros((Ny, Nu))
        for j in range(Ny):
            cos_vy = np.cos(2 * np.pi * self.v * self.y[j])
            sin_vy = np.sin(2 * np.pi * self.v * self.y[j])
            # Re{F*e^(jvy)} = Re*cos(vy) - Im*sin(vy)
            tc = self.real * cos_vy[:, np.newaxis] - self.imag * sin_vy[:, np.newaxis]
            # Im{F*e^(jvy)} = -Re*sin(vy) - Im*cos(vy)
            ts = -self.real * sin_vy[:, np.newaxis] - self.imag * cos_vy[:, np.newaxis]
            F_c[j, :] = np.trapezoid(tc, self.v, axis=0)
            F_s[j, :] = np.trapezoid(ts, self.v, axis=0)

        # Stage 2: for each x, integrate over u
        image = np.zeros((Ny, Nx))
        for i in range(Nx):
            cos_ux = np.cos(2 * np.pi * self.u * self.x[i])
            sin_ux = np.sin(2 * np.pi * self.u * self.x[i])
            term = F_c * cos_ux[np.newaxis, :] + F_s * sin_ux[np.newaxis, :]
            image[:, i] = np.trapezoid(term, self.u, axis=1)

        return image
```

---

## 10. Common Mistakes & Exam Checklist

### Fatal Mistakes

| Mistake | Consequence |
|---|---|
| `np.fft` anywhere in code | **Strictly prohibited** |
| O(N^4) nested loops for 2D-CFT | Won't finish in time |
| Forgetting negative harmonics | Wrong FS reconstruction |
| No `np.unwrap()` in phase MSE | Phase jumps inflate error |
| Using `self.x/self.y` as CFT2D freq axes | Wrong spectrum, no edge map |
| Hardcoded Sobel/Canny filters | Must come from CFT pipeline |
| `self.T = t[-1]` (wrong) vs `t[-1]-t[0]` | Wrong omega, wrong coeffs |

### Mental Checklist

**Step 1 — identify property:**
- Has `d/dt`? → Differentiation: multiply X(f) by `(j*2*pi*f)^n`
- Has `x(t-t0)`? → Time-shift: multiply X(f) by `e^(-j*2*pi*f*t0)`
- Has `x(a*t)`? → Scaling: `(1/|a|) * X(f/a)`
- Has both scale AND modulation `e^(j*2*pi*f0*t)`? → `(1/|a|) * X((f-f0)/a)`

**Step 2 — set axes:**
- `t` range: wide (`-20:20`) for sinusoids, narrow (`-5:5`) for Gaussian/rect
- `f` range: `freq_Hz = omega/(2*pi)`. E.g. `omega=4 -> f~0.64 Hz`

**Step 3 — phase threshold:**
```python
phase = np.where(np.abs(Y) > 1e-2, np.angle(Y), 0)
```

**Step 4 — MSE:**
```python
MSE = np.mean((measured - theoretical)**2)
```

**Step 5 — phase unwrap for MSE:**
```python
valid = np.abs(Y_theory) > 1e-3
MSE_phase = np.mean((np.unwrap(np.angle(Y_meas))[valid] - np.unwrap(np.angle(Y_theory))[valid])**2)
```

### Quick Checklist

```
BEFORE CODING:
[ ] Identify the CFT property being tested
[ ] Note all parameters (a, t0, f0, omega, N, etc.)
[ ] Set t range and f range appropriately

CODING:
[ ] SignalGenerator with all needed signal methods
[ ] CFTAnalyzer.compute_cft() using np.trapezoid — no np.fft
[ ] Compute numerical CFT of modified/derived signal
[ ] Compute property-based theoretical CFT
[ ] Plot magnitude (blue=numerical, red-dashed=theoretical)
[ ] Plot phase with threshold applied
[ ] Compute and print magnitude MSE
[ ] Compute and print phase MSE (with np.unwrap + mask)
[ ] Print confirmation if property is verified

FOR 2D CFT (offline Task 2):
[ ] CFT Stage 1: loop over u, integrate over x (axis=1) -> Ix_c, Ix_s
[ ] CFT Stage 2: loop over v, integrate over y (axis=0) -> real_F, imag_F
[ ] ICFT Stage 1: loop over y, integrate F over v (axis=0) -> F_c, F_s
[ ] ICFT Stage 2: loop over x, integrate over u (axis=1) -> image
[ ] Use self.u / self.v NOT self.x / self.y as freq axes
[ ] No np.fft anywhere

SUBMISSION:
[ ] Online: single file StudentID.py
[ ] Offline: StudentID.zip containing fs_redrawer.py + cft_edge_detector.py
[ ] No np.fft or scipy.fft anywhere in code
```

---

*Good luck tomorrow! Workflow: identify property -> numerical CFT -> property-based CFT -> plot both -> compute MSE*
