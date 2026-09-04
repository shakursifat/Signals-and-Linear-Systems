# CSE 220 — Signals and Linear Systems
## Offline 3: DFT & FFT — Theory, Implementation & Analysis

> **Due:** Saturday, 5 September 2026  
> **Files written:** `transforms.py`, `bigmul.py`, `image_conv.py`

---

## Table of Contents

1. [Core Theory](#1-core-theory)
2. [transforms.py — The Shared Core](#2-transformspy)
3. [Task A — Big-Integer Multiplication](#3-task-a)
4. [Task B — Image Blur via 2D Convolution](#4-task-b)
5. [Bugs Found & Fixed](#5-bugs-found--fixed)
6. [Runtime Complexity Analysis](#6-runtime-complexity-analysis)
7. [Additional Important Concepts](#7-additional-important-concepts)

---

## 1. Core Theory

### 1.1 The Discrete Fourier Transform (DFT)

The DFT converts a length-N time-domain sequence into N complex frequency bins.

**Analysis (forward transform):**

```
X[k] = sum_{n=0}^{N-1}  x[n] * exp(-2*pi*j*k*n / N),   k = 0..N-1
```

**Synthesis (inverse transform):**

```
x[n] = (1/N) * sum_{k=0}^{N-1}  X[k] * exp(+2*pi*j*k*n / N),   n = 0..N-1
```

**Key properties:**

| Property    | Formula                                                  |
|-------------|----------------------------------------------------------|
| Linearity   | DFT{ax + by} = a·DFT{x} + b·DFT{y}                     |
| Time shift  | delay by m ⟹ multiply spectrum by exp(-2πjkm/N)         |
| Convolution | DFT{x ★ h} = DFT{x} · DFT{h}  (pointwise product)      |
| Parseval    | Σ|x[n]|² = (1/N)·Σ|X[k]|²                              |

**Complexity:** O(N²) — each of N output bins sums N terms.

---

### 1.2 The Fast Fourier Transform (FFT)

The **Cooley-Tukey Radix-2 DIT FFT** reduces O(N²) to **O(N log₂ N)** by
recursively splitting into two half-size DFTs.

**Butterfly identity:**

```
X[k]         = E[k] + W_N^k · O[k]
X[k + N/2]   = E[k] - W_N^k · O[k]

W_N^k = exp(-2πj·k/N)  (twiddle factor)
E[k], O[k] = DFTs of even/odd sub-sequences
```

**Our iterative steps:**
1. Bit-reversal permutation — reorder input for in-place DIT butterflies
2. log₂(N) butterfly stages — each stage doubles the sub-DFT size being merged
3. Twiddle factors computed **once per stage**, not once per butterfly

**Inverse FFT via conjugate trick (reuses the forward butterfly):**

```
IFFT(X) = conj(FFT(conj(X))) / N
```

---

### 1.3 The Convolution Theorem

```
a ★ c  =  IDFT( DFT{a} · DFT{c} )       (★ = convolution, · = pointwise ×)
```

**Full pipeline:**
```
a, c  →  zero-pad  →  FFT  →  A, C  →  A×C pointwise  →  IFFT  →  a★c
```

Why it works: complex exponentials are eigenfunctions of the shift operator.
In the frequency domain each bin is scaled independently, so convolution
decouples into N scalar multiplications.

---

### 1.4 Circular vs Linear Convolution — The Padding Rule

A length-N DFT gives **circular convolution** (overflow wraps back):

```
(a ⊛ c)[m] = Σ a[i]·c[(m-i) mod N]
```

We need **linear convolution** (no wrap). For inputs of length n and q the
linear result has **n + q − 1** entries.

> **Padding Rule:** zero-pad both inputs to N ≥ n+q−1.  
> For radix-2 FFT pad further to N = next_power_of_two(n+q−1).

**Example:**

```python
a = [1, 2]   # 1 + 2x
c = [3, 4]   # 3 + 4x
# Linear result:            [3, 10, 8]
# N=2 (no padding):  [3+8, 4+6] = [11, 10]   ← WRONG (wrap-around)
# N=4 (padded):      [3, 10, 8, 0]            ← CORRECT
```

---

## 2. transforms.py

### 2.1 next_power_of_two

```python
def next_power_of_two(n):
    if n <= 1:
        return 1
    result = 1
    while result < n:
        result <<= 1      # multiply by 2 via bit-shift
    return result

# next_power_of_two(5)=8   next_power_of_two(64)=64   next_power_of_two(65)=128
```

---

### 2.2 DFTAnalyzer — Naive O(N²)

Builds the full N×N twiddle-factor matrix using NumPy's outer product:

```python
class DFTAnalyzer:
    name = "dft"

    def transform(self, x):
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        k = np.arange(N, dtype=np.float64)
        n = np.arange(N, dtype=np.float64)
        W = np.exp(-2j * np.pi * np.outer(k, n) / N)   # W[k,n] = e^{-2πjkn/N}
        return W @ x          # O(N²) matrix–vector multiply via BLAS

    def inverse(self, X):
        X = np.asarray(X, dtype=np.complex128)
        N = len(X)
        k = np.arange(N, dtype=np.float64)
        n = np.arange(N, dtype=np.float64)
        W_inv = np.exp(2j * np.pi * np.outer(n, k) / N)
        return (W_inv @ X) / N
```

Why the matrix approach? NumPy's `@` calls optimised BLAS — much faster than
an explicit Python double loop, yet mathematically identical to the sum.

---

### 2.3 FFTTransformer — Radix-2 Cooley-Tukey

#### Step 1 — Bit-Reversal Permutation

| Original | Binary | Bit-reversed | New |
|----------|--------|--------------|-----|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

```python
log2N = int(round(np.log2(N)))
rev = np.zeros(N, dtype=np.int64)
tmp = np.arange(N, dtype=np.int64)
for _ in range(log2N):
    rev = (rev << 1) | (tmp & 1)
    tmp >>= 1
x = x[rev].copy()
```

#### Step 2 — Vectorised Butterfly Stages

```python
length = 2
while length <= N:
    half = length // 2

    # Twiddle factors computed ONCE per stage
    m = np.arange(half, dtype=np.float64)
    twiddle = np.exp(-2j * np.pi * m / length)     # shape (half,)

    # Process ALL groups simultaneously via reshape
    xs = x.reshape(-1, length)                      # (N/length, length)
    u = xs[:, :half].copy()                         # .copy() is essential!
    v = xs[:, half:] * twiddle[np.newaxis, :]
    xs[:, :half] = u + v
    xs[:, half:] = u - v

    length <<= 1
```

> **Why `.copy()`?** Without it `u` is a NumPy *view* of `xs`. Writing
> `xs[:, :half] = u+v` immediately corrupts `u`, so the second line reads
> wrong values. This was **Bug #1**.

#### Inverse via Conjugate Trick

```python
def inverse(self, X):
    x = self._fft_core(np.conj(X))
    return np.conj(x) / N
```

Reuses the exact same butterfly machinery — no second implementation needed.

#### Quick Self-Test

```python
x = np.random.randn(64) + 1j*np.random.randn(64)
d, f = DFTAnalyzer(), FFTTransformer()
assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9   # agree
assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9     # round-trip
```

---

### 2.4 ArbitraryLengthFFT — Bluestein's Chirp-Z (Bonus)

Handles any N in O(N log N) using the algebraic identity:

```
k·n = k²/2 + n²/2 − (k−n)²/2
```

This rewrites the DFT sum as a linear convolution of chirp sequences,
computed with a radix-2 FFT of padded length M ≥ 2N−1:

```python
def _bluestein(self, x):
    N = len(x)
    if N & (N-1) == 0:                          # power of two → use radix-2
        return self._radix2(x.copy())
    n  = np.arange(N, dtype=np.float64)
    w  = np.exp(-1j * np.pi * n * n / N)        # chirp
    a  = x * w
    M  = next_power_of_two(2*N - 1)
    b  = np.zeros(M, dtype=np.complex128)
    b[:N]        = np.conj(w)
    b[M-N+1:]    = np.conj(w[1:])
    A   = self._radix2(np.pad(a, (0, M-N)))
    B   = self._radix2(b)
    out = np.conj(self._radix2(np.conj(A * B))) / M     # IFFT trick
    return w * out[:N]
```

---

## 3. Task A — Big-Integer Multiplication

### 3.1 Core Idea: Digits as Polynomial Coefficients

```
A = a₀ + a₁B + a₂B² + ...     (little-endian limbs, BASE B = 10^BASE_DIGITS)
A·C  =  polynomial product  =  linear convolution of limb arrays
```

After convolution, a carry sweep turns each oversized coefficient into a
proper limb.

**Why BASE = 10⁴ (BASE_DIGITS = 4)?**

The largest convolution coefficient is p_max ≤ n·(B−1)².  
A 64-bit double has a 53-bit mantissa → exact integers up to 2^53 ≈ 9×10¹⁵.

| Base B | (B-1)² | With n=10⁶ limbs | Safe? |
|--------|--------|-------------------|-------|
| 10⁴    | ~10⁸   | p_max ~10¹⁴       | ✅ yes |
| 10⁹    | ~10¹⁸  | p_max ~10²⁴       | ❌ corrupt! |

---

### 3.2 to_limbs

```python
def to_limbs(text, base_digits=BASE_DIGITS):
    # 1. Extract sign
    if text.startswith('-'):
        sign, digits = -1, text[1:]
    else:
        sign, digits = 1, text.lstrip('+')

    digits = digits.lstrip('0') or '0'

    # 2. Left-pad so length is a multiple of base_digits
    rem = len(digits) % base_digits
    if rem:
        digits = '0' * (base_digits - rem) + digits

    # 3. Split into groups of base_digits, reverse → little-endian
    limbs = []
    for i in range(len(digits) - base_digits, -1, -base_digits):
        limbs.append(int(digits[i:i+base_digits]))

    return sign, np.array(limbs, dtype=np.int64)

# "123456789" → sign=1, limbs=[6789, 2345, 1]
#   = 1·B² + 2345·B¹ + 6789·B⁰
```

---

### 3.3 from_limbs

```python
def from_limbs(sign, limbs, base_digits=BASE_DIGITS):
    base = 10 ** base_digits
    L = [int(v) for v in limbs]

    # Carry sweep (LSB → MSB)
    carry = 0
    for i in range(len(L)):
        total   = L[i] + carry
        L[i]    = total % base
        carry   = total // base
    while carry:
        L.append(carry % base); carry //= base

    # Strip trailing zeros in little-endian (= leading zeros in number)
    while len(L) > 1 and L[-1] == 0:
        L.pop()

    # Build decimal: MSB first, inner limbs zero-padded to base_digits
    result = str(L[-1])
    for i in range(len(L)-2, -1, -1):
        result += str(L[i]).zfill(base_digits)   # zfill is critical!

    return ('-' + result if sign == -1 and result != '0' else result)
```

> **Critical:** `.zfill(base_digits)` zero-pads inner limbs.  
> Without it, limb 42 prints as "42" not "0042", silently corrupting digits.

---

### 3.4 multiply_transform

```python
def multiply_transform(a, b, engine):
    n, q = len(a), len(b)
    linear_len = n + q - 1                     # minimum safe length

    if engine.name == "arbitrary":
        N = linear_len                          # Bluestein handles any N
    else:
        N = next_power_of_two(linear_len)       # DFT and FFT both use 2^k

    a_pad = np.zeros(N, dtype=np.complex128); a_pad[:n] = a
    b_pad = np.zeros(N, dtype=np.complex128); b_pad[:q] = b

    A = engine.transform(a_pad)
    B = engine.transform(b_pad)
    c = engine.inverse(A * B)                  # pointwise × in freq domain

    return np.round(c.real).astype(np.int64), N   # imaginary part is noise
```

> **Bug #2 fixed:** DFT originally used `N = linear_len` instead of
> `next_power_of_two(linear_len)`, causing the reported N to differ from
> the expected output.

---

### 3.5 multiply_schoolbook (Optional Baseline O(n²))

```python
def multiply_schoolbook(a, b):
    result = np.zeros(len(a)+len(b)-1, dtype=np.int64)
    for i in range(len(a)):
        result[i:i+len(b)] += a[i] * b    # one NumPy vector-add per row
    return result
```

NumPy's small constant factor makes this faster than FFT for small n.

---

### 3.6 Expected Report Format

```
Task A -- big-integer multiplication by spectral convolution
input file          : inputs/1.txt
method              : dft
digits of A / B     : 12 / 9
base                : 10^4
limbs of A / B      : 3 / 3
transform length N  : 8
digits of product   : 21
verification        : MATCH
```

Verification (only allowed Python big-int call):
```python
verdict = "MATCH" if product_str == str(int(text_a) * int(text_b)) else "MISMATCH"
```

---

## 4. Task B — Image Blur via 2D Convolution

### 4.1 2D DFT by Separability

The 2D DFT exponential factorises:

```
exp(-2πj(ur/P + vc/Q)) = exp(-2πj·ur/P) × exp(-2πj·vc/Q)
```

So: **transform every row**, then **transform every column** of the result.

| Method          | Cost for N×N image |
|-----------------|--------------------|
| Direct 2D DFT   | O(N⁴)             |
| Separable DFT   | O(N³)             |
| Separable FFT   | O(N² log N)       |

---

### 4.2 transform_2d & inverse_2d

```python
def transform_2d(plane, engine):
    result = np.empty(plane.shape, dtype=np.complex128)
    for r in range(plane.shape[0]):       # Pass 1: every row
        result[r, :] = engine.transform(plane[r, :])
    for c in range(plane.shape[1]):       # Pass 2: every column
        result[:, c] = engine.transform(result[:, c])
    return result

def inverse_2d(spectrum, engine):
    result = np.empty(spectrum.shape, dtype=np.complex128)
    for c in range(spectrum.shape[1]):    # Inverse columns
        result[:, c] = engine.inverse(spectrum[:, c])
    for r in range(spectrum.shape[0]):    # Inverse rows
        result[r, :] = engine.inverse(result[r, :])
    return result
```

---

### 4.3 convolve_plane — Linear & Circular

#### Linear Convolution (zero-padded, `circular=False`)

```python
def convolve_plane(plane, kernel, engine, circular=False):
    H, W   = plane.shape
    kh, kw = kernel.shape

    if not circular:
        full_h = H + kh - 1
        full_w = W + kw - 1

        if engine.name == "fft":
            pad_h = next_power_of_two(full_h)
            pad_w = next_power_of_two(full_w)
        else:
            pad_h, pad_w = full_h, full_w

        plane_pad  = np.zeros((pad_h, pad_w)); plane_pad[:H, :W] = plane
        kernel_pad = np.zeros((pad_h, pad_w)); kernel_pad[:kh, :kw] = kernel   # ORIGIN!

        F = transform_2d(plane_pad, engine) * transform_2d(kernel_pad, engine)
        out = inverse_2d(F, engine).real

        # Crop at (kh//2, kw//2) — compensates for kernel-at-origin placement
        r0, c0 = kh//2, kw//2
        return out[r0:r0+H, c0:c0+W]
```

> **Kernel placement:** the kernel sits at origin (0,0), NOT centred.
> If centred, the output shifts diagonally. The crop at `(kh//2, kw//2)` fixes it.

#### Circular Convolution (no padding, `circular=True`)

```python
    else:
        # Wrap kernel using modular indexing (implements CIFT from origin)
        kpad = np.zeros((H, W), dtype=np.float64)
        for ki in range(kh):
            for kj in range(kw):
                kpad[(ki - kh//2) % H, (kj - kw//2) % W] += kernel[ki, kj]

        F = transform_2d(plane, engine) * transform_2d(kpad, engine)
        return inverse_2d(F, engine).real
```

Content that "falls off" one edge reappears on the opposite edge —
the deliberate wraparound artefact shown in `wraparound.png`.

---

### 4.4 convolve_plane_direct — The Correctness Oracle

```python
def convolve_plane_direct(plane, kernel):
    H, W   = plane.shape
    kh, kw = kernel.shape
    out = np.zeros((H, W), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            v = 0.0
            for i in range(kh):
                for j in range(kw):
                    pr = r + kh//2 - i
                    pc = c + kw//2 - j
                    if 0 <= pr < H and 0 <= pc < W:
                        v += plane[pr, pc] * kernel[i, j]
            out[r, c] = v
    return out
```

Four nested loops, zero NumPy vectorisation — slow on purpose and obviously
correct. Only used on the top-left 64×64 corner:

```python
error = np.max(np.abs(convolve_plane(corner, kernel, engine)
                      - convolve_plane_direct(corner, kernel)))
# Expect ~1e-15; anything > 1e-9 is a code bug, not rounding
```

---

### 4.5 Expected Report & Image Labels

**report.txt:**
```
Task B -- 2D convolution through the frequency domain
image               : images/skyline512.png  (512 x 512, RGB)
kernel              : bokeh  (19 x 19)
engine              : fft
linear-conv size    : 530 x 530
transform size      : 1024 x 1024
max |spectral - direct| on 64x64 crop : 3.220e-15
verification        : MATCH
```

**comparison.png panel labels (exact strings):**
- `"original"`
- `"linear convolution (zero-padded)"`
- `"circular convolution (no padding)"`

**suptitle:** `"skyline512.png, bokeh kernel 19x19, engine=fft"`

---

## 5. Bugs Found & Fixed

### Bug 1 — FFT butterfly corrupted by NumPy view aliasing

```python
# BROKEN: u is a VIEW of x, not a copy
for start in range(0, N, length):
    u = x[start : start+half]
    v = x[start+half : start+length] * twiddle
    x[start : start+half] = u + v        # mutates x -> corrupts u instantly!
    x[start+half : start+length] = u - v # u already changed -> wrong!

# FIXED: reshape does all groups at once, .copy() breaks aliasing
xs = x.reshape(-1, length)
u = xs[:, :half].copy()                  # true copy
v = xs[:, half:] * twiddle[np.newaxis, :]
xs[:, :half] = u + v
xs[:, half:] = u - v
```

---

### Bug 2 — DFT reported wrong transform length N

```python
# BROKEN: DFT used raw length -> N=5 for 3+3-1=5 limbs
N = linear_len if engine.name == "dft" else next_power_of_two(linear_len)

# FIXED: both DFT and FFT pad to next power of two
N = next_power_of_two(linear_len)   # (except ArbitraryLengthFFT)
```

Expected output shows N=8, not N=5, for a 3+3-1=5 element convolution.

---

### Bug 3 — Wrong engine for inputs 1 & 2

PDF specifies inputs 1 & 2 → `--engine dft`, inputs 3 & 4 → `--engine fft`.
Initially ran input 1 with `--engine fft`. Re-ran with correct flags.

---

### Bug 4 — Task A report.txt field names wrong

| Was                                   | Expected                                              |
|---------------------------------------|-------------------------------------------------------|
| `Task A -- single multiplication`     | `Task A -- big-integer multiplication by spectral convolution` |
| `engine       : dft`                  | `method              : dft`                           |
| `operand A    : 12 digits`            | `digits of A / B     : 12 / 9`                       |
| `base         : 10^4 = 10000`         | `base                : 10^4`                          |
| `limbs A : 3` + `limbs B : 3` (separate lines) | `limbs of A / B      : 3 / 3`              |
| `transform N  : 8`                    | `transform length N  : 8`                             |
| `product      : 21 digits`            | `digits of product   : 21`                            |

---

### Bug 5 — Task B report.txt field names wrong

| Was                                          | Expected                                            |
|----------------------------------------------|-----------------------------------------------------|
| `Task B -- single image blur`                | `Task B -- 2D convolution through the frequency domain` |
| `image path / image size / color mode` (3 lines) | `image : path  (H x W, RGB)` (1 line)          |
| `verification (64x64 corner) : OK (3.22e-15)` | `max |spectral - direct| on 64x64 crop : 3.220e-15` |

---

### Bug 6 — comparison.png labels wrong

| Was                                    | Expected                                       |
|----------------------------------------|------------------------------------------------|
| `blurred (linear)`                     | `linear convolution (zero-padded)`             |
| `wraparound (circular)`                | `circular convolution (no padding)`            |
| `skyline512.png -- bokeh kernel, ...`  | `skyline512.png, bokeh kernel 19x19, ...`      |

---

## 6. Runtime Complexity Analysis

### Task A

| Method       | Complexity  | Time @ 4096 digits |
|--------------|-------------|-------------------|
| Schoolbook   | O(n²)       | 0.0027 s          |
| Naive DFT    | O(n²)       | 0.3676 s          |
| Radix-2 FFT  | O(n log n)  | 0.0015 s          |

DFT and Schoolbook are both O(n²) but DFT has a larger constant (matrix ops).
FFT dominates at large n.

### Task B

| Method          | Cost for N×N image, K×K kernel |
|-----------------|--------------------------------|
| Direct spatial  | O(N² K²)                      |
| Separable DFT   | O(N³)                         |
| Separable FFT   | O(N² log N)                   |

**Key insight:** FFT cost is **independent of kernel size K**. Both image and
kernel pad to the same transform size, so increasing K has no effect on the
FFT curve — it stays flat on the growing-kernel benchmark.

### Slope Verification on Log-Log Plot

| Method      | Our measured slope | Expected    |
|-------------|--------------------|-------------|
| Naive DFT   | ~2.56 → 3.0 at large N | O(N³)  |
| Radix-2 FFT | ~1.40              | O(N² log N) |
| Direct      | ~1.98              | O(N²)       |

Slopes match expected complexity — the implementation is correct.

---

## 7. Additional Important Concepts

### 7.1 Zero-Phase Kernel Placement

When convolving via FFT, place the kernel at **origin (0,0)** of the padded
array, NOT centred. Centring displaces the output diagonally. Crop at
`(kh//2, kw//2)` to undo the origin-offset displacement.

```python
kernel_padded[:kh, :kw] = kernel          # kernel at origin
result = full[kh//2 : kh//2+H, kw//2 : kw//2+W]   # compensating crop
```

---

### 7.2 Number Theoretic Transform (NTT) — Bonus +10

Replace complex exponentials with powers of a primitive root modulo a prime.  
For p = 998244353 = 119·2²³+1, primitive root g = 3:

```
X[k] = Σ x[n]·g^(kn)  mod p
```

- All arithmetic is exact integers — no floating-point rounding.
- Every output coefficient must be < p, so limb base must be smaller.
- With p ≈ 10⁹, need B ≪ √p ≈ 31623.

---

### 7.3 The Gibbs Phenomenon

A box (rectangular) kernel has a sharp frequency cutoff. The Fourier series of
a sharp step takes infinitely many terms to converge, and truncating it causes
ringing oscillations near discontinuities — the **Gibbs phenomenon**.
Gaussian and bokeh kernels have smooth spectra, producing far less ringing.

---

### 7.4 FFT vs Direct — Why the Kernel Size Doesn't Matter

For a 512×512 image with a 41×41 kernel:

- Direct: 512² × 41² ≈ **4.4 billion** multiplications
- FFT (pad to 1024×1024): ~21 million operations — **~850× speedup**

As the kernel grows, direct spatial cost rises as K². FFT cost is constant
because the transform size is fixed by the image dimensions.

---

### 7.5 Floating-Point Precision & the Rounding Step

After `c = IFFT(FFT(a) × FFT(b))` the results should be integers, but
double-precision errors shift them fractionally:

```python
result = np.round(c.real).astype(np.int64)   # imaginary part is pure noise
```

Safe because our BASE = 10⁴ keeps every convolution coefficient well below
the 2^53 mantissa limit of a 64-bit double.

---

### 7.6 Why Benchmark Graphs Differ Between Machines

> *"The timings come from one particular machine. Your absolute numbers will*
> *differ; the shapes of the curves should not."* — PDF specification

- Absolute times scale with CPU speed and cache (10–100× differences possible).
- Which curves hit the 8s time budget changes → different visible data points.
- **What matters is the slope on the log-log plot:**
  - slope ~1 = O(n log n)
  - slope ~2 = O(n²)
  - slope ~3 = O(n³)

---

### 7.7 Spectral Leakage

The DFT assumes the signal repeats with period N. If it doesn't, the
boundary discontinuity spreads energy into neighbouring bins — **spectral
leakage**. In image convolution this appears as the wraparound artefact
in `wraparound.png`: content from one edge bleeds to the opposite edge
because the circular transform "sees" the image repeating.

---

### 7.8 Mixed-Radix Cooley-Tukey vs Bluestein

| Method | Handles arbitrary N? | Complexity |
|--------|---------------------|------------|
| Radix-2 Cooley-Tukey | Only powers of 2 | O(N log N) |
| Mixed-radix Cooley-Tukey | N with small prime factors | O(N log N) |
| Bluestein chirp-z | Any N, always | O(N log N) |

Bluestein is more robust for arbitrary N because it doesn't depend on
N's prime factorisation.

---

### 7.9 Windowing Functions

For **spectrum estimation** (not convolution), one applies a window function
(Hann, Hamming, Blackman) before the DFT to reduce spectral leakage by
tapering the signal to zero at the boundaries.

For convolution, zero-padding is correct and sufficient — windowing would
distort the kernel and corrupt the result.

---

## Commands Run

```bash
# Task A: big-integer multiplication
python bigmul.py inputs/1.txt --engine dft --out-dir outputs/task_a/1
python bigmul.py inputs/2.txt --engine dft --out-dir outputs/task_a/2
python bigmul.py inputs/3.txt --engine fft --out-dir outputs/task_a/3
python bigmul.py inputs/4.txt --engine fft --out-dir outputs/task_a/4
python bigmul.py --benchmark --out-dir outputs/task_a/benchmark

# Task B: image blur via 2D convolution
python image_conv.py images/skyline512.png --kernel bokeh --param 9 --engine fft --out-dir outputs/task_b/skyline_bokeh
python image_conv.py images/sunset512.png --gray --kernel motion --param 41 --engine fft --out-dir outputs/task_b/sunset_motion
python image_conv.py images/skyline256.png --gray --kernel gaussian --param 21 --engine dft --out-dir outputs/task_b/skyline256_gaussian_dft
python image_conv.py images/skyline512.png --benchmark --out-dir outputs/task_b/benchmark
```

---

*Verification: Task A inputs 1–4 → all **MATCH**.*  
*Task B 64×64 corner errors: 3.22e-15, 3.33e-16, 1.62e-14 — all far below the 1e-9 threshold.*
