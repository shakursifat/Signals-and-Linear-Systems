# Extra DFT / FFT Functions — Theory & Implementation

> **Context:** These functions extend the transforms covered in `transforms.py`, `bigmul.py` (Task A), and `image_conv.py` (Task B).  
> They are **standalone and importable** — none of them call `numpy.fft`, `scipy.fft`, `numpy.convolve`, or any library transform.

---

## Table of Contents

1. [Utility — Power-of-Two Helpers](#1-utility--power-of-two-helpers)
2. [Frequency Analysis](#2-frequency-analysis)
3. [Convolution Variants](#3-convolution-variants)
4. [Signal Reconstruction](#4-signal-reconstruction)
5. [Spectral Properties](#5-spectral-properties)
6. [2D Transform Utilities (Task B)](#6-2d-transform-utilities-task-b)
7. [Polynomial / Big-Integer Helpers (Task A)](#7-polynomial--big-integer-helpers-task-a)
8. [Windowing Functions](#8-windowing-functions)
9. [Correlation](#9-correlation)
10. [Number Theoretic Transform (NTT) — Bonus](#10-number-theoretic-transform-ntt--bonus)

---

## 1. Utility — Power-of-Two Helpers

### 1.1 `is_power_of_two(n)`

```python
def is_power_of_two(n: int) -> bool:
    """
    Return True iff n is a positive integer that is an exact power of two.

    The radix-2 FFT (FFTTransformer._fft_core) raises ValueError unless its
    input has power-of-two length. This guard lets callers cheaply check
    before calling transform().

    Algorithm: the bit-twiddling trick  n & (n-1) == 0  clears the lowest set
    bit of n.  For a power of two, only one bit is set, so the result is 0.

    Examples
    --------
    is_power_of_two(1)   -> True   (2^0)
    is_power_of_two(8)   -> True   (2^3)
    is_power_of_two(6)   -> False
    is_power_of_two(0)   -> False  (0 has no set bits, but is not a power of two)
    """
    return n > 0 and (n & (n - 1)) == 0
```

**Why it matters:** `FFTTransformer.transform` already guards with the same
test, but calling it yourself before zero-padding avoids the `ValueError` and
makes the padding logic self-documenting.

---

### 1.2 `pad_to_power_of_two(x)`

```python
import numpy as np

def pad_to_power_of_two(x: np.ndarray) -> np.ndarray:
    """
    Zero-pad 1D array x to the next power-of-two length and return the copy.

    Used by both Task A (multiply_transform) and Task B (convolve_plane) to
    ensure FFTTransformer.transform never sees a non-power-of-two length.

    Parameters
    ----------
    x : 1D numpy array (real or complex)

    Returns
    -------
    numpy.ndarray, dtype=complex128, length = next_power_of_two(len(x))

    Example
    -------
    pad_to_power_of_two(np.array([1, 2, 3]))
    -> array([1.+0.j, 2.+0.j, 3.+0.j, 0.+0.j])   # padded from 3 to 4
    """
    from transforms import next_power_of_two
    x = np.asarray(x, dtype=np.complex128)
    N = next_power_of_two(len(x))
    padded = np.zeros(N, dtype=np.complex128)
    padded[:len(x)] = x
    return padded
```

---

### 1.3 `log2_exact(n)`

```python
def log2_exact(n: int) -> int:
    """
    Return log2(n) as an integer, assuming n is a positive power of two.

    Used internally by FFTTransformer._fft_core to compute the number of
    butterfly stages: log2N = int(round(np.log2(N))).
    This function provides a bit-manipulation alternative with no
    floating-point involved.

    Parameters
    ----------
    n : int, must be a power of two and >= 1

    Returns
    -------
    int -- the exact base-2 logarithm

    Raises
    ------
    ValueError if n is not a power of two

    Example
    -------
    log2_exact(8)  -> 3
    log2_exact(1)  -> 0
    """
    if not is_power_of_two(n):
        raise ValueError(f"log2_exact: {n} is not a power of two")
    p = 0
    m = n
    while m > 1:
        m >>= 1
        p += 1
    return p
```

---

## 2. Frequency Analysis

### 2.1 `dft_frequency_bins(N, fs=1.0)`

```python
def dft_frequency_bins(N: int, fs: float = 1.0) -> np.ndarray:
    """
    Return the physical frequencies corresponding to DFT output bins 0..N-1.

    The DFT output X[k] corresponds to a sinusoid of frequency k/N cycles
    per sample. Multiplying by the sample rate ``fs`` gives cycles per second
    (Hz).

    The spectrum is naturally ordered 0, df, 2*df, ..., (N-1)*df.
    Bins above N//2 represent negative frequencies due to the periodicity of
    the DFT:
        k > N//2  ->  equivalent negative frequency  (k - N) * df

    Parameters
    ----------
    N  : int  -- transform length
    fs : float -- sample rate in Hz (default 1.0 -> normalised frequencies)

    Returns
    -------
    numpy.ndarray, shape (N,), dtype float64
        Frequencies in Hz (or in cycles/sample if fs=1).

    Example (N=8, fs=8 Hz)
    -----------------------
    dft_frequency_bins(8, fs=8)
    -> [0., 1., 2., 3., 4., -3., -2., -1.]   # standard DFT order
    """
    k = np.arange(N, dtype=np.float64)
    freqs = k * fs / N
    freqs[N // 2 + 1:] -= fs   # fold upper half to negative frequencies
    return freqs
```

**Connection to Task B:** When inspecting the 2D Fourier spectrum of an image
(e.g. with `transform_2d`) the DC component sits at (0, 0); the spatial
frequencies increase outward in the un-shifted spectrum.

---

### 2.2 `fftshift_1d(X)`

```python
def fftshift_1d(X: np.ndarray) -> np.ndarray:
    """
    Shift the zero-frequency component to the centre of the spectrum.

    The standard DFT layout puts DC at index 0. For visualisation it is more
    natural to see:
        ... -2*df  -df  0  +df  +2*df ...

    This is the manual equivalent of numpy.fft.fftshift.
    It works by rotating the array by N//2 positions.

    Parameters
    ----------
    X : 1D numpy array of complex (DFT output, length N)

    Returns
    -------
    numpy.ndarray -- same dtype, length N, DC centred at index N//2
    """
    N = len(X)
    shift = N // 2
    return np.roll(X, shift)
```

---

### 2.3 `power_spectrum(X)`

```python
def power_spectrum(X: np.ndarray) -> np.ndarray:
    """
    Compute the one-sided power spectral density from a DFT/FFT output.

    Power in bin k is  |X[k]|^2 / N.  For a real input signal, the spectrum
    is Hermitian-symmetric: X[N-k] = conj(X[k]).  The one-sided spectrum
    therefore keeps only bins 0..N//2 and doubles bins 1..N//2-1 to
    conserve total power (Parseval's theorem).

    Parameters
    ----------
    X : 1D numpy array -- DFT output (complex128), length N

    Returns
    -------
    numpy.ndarray of float64, length N//2 + 1
        One-sided power, normalised so that sum equals mean square of input.

    Theory (Parseval)
    -----------------
    sum|x[n]|^2 = (1/N) * sum|X[k]|^2
    """
    N = len(X)
    psd = (np.abs(X) ** 2) / N
    one_sided = psd[:N // 2 + 1].copy()
    one_sided[1:N // 2] *= 2   # double interior bins for energy from negative half
    return one_sided
```

---

### 2.4 `dominant_frequency(x, fs=1.0, engine=None)`

```python
def dominant_frequency(x: np.ndarray, fs: float = 1.0, engine=None) -> float:
    """
    Find the frequency with the highest power in signal x.

    Parameters
    ----------
    x      : 1D real-valued numpy array
    fs     : sample rate in Hz
    engine : DFTAnalyzer or FFTTransformer instance (defaults to DFTAnalyzer)

    Returns
    -------
    float -- the dominant frequency in Hz

    Example
    -------
    t = np.linspace(0, 1, 64, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)   # 5 Hz sinusoid
    dominant_frequency(x, fs=64)     -> 5.0
    """
    from transforms import DFTAnalyzer
    if engine is None:
        engine = DFTAnalyzer()
    X = engine.transform(x.astype(np.complex128))
    N = len(x)
    magnitudes = np.abs(X[:N // 2 + 1])
    k_peak = int(np.argmax(magnitudes))
    return k_peak * fs / N
```

---

## 3. Convolution Variants

### 3.1 `linear_convolve(a, b, engine)`

```python
def linear_convolve(a: np.ndarray, b: np.ndarray, engine) -> np.ndarray:
    """
    Linear (aperiodic) convolution of two 1D sequences via the frequency domain.

    This is the core operation in Task A: multiplying polynomial coefficients
    (limbs) is identical to linearly convolving two sequences.

    Steps (same as multiply_transform in bigmul.py):
      1.  linear_len = len(a) + len(b) - 1
      2.  N = next_power_of_two(linear_len)
      3.  Zero-pad a and b to length N.
      4.  A = engine.transform(a_pad), B = engine.transform(b_pad)
      5.  c = engine.inverse(A * B)
      6.  Return c[:linear_len].real

    Parameters
    ----------
    a, b   : 1D numpy arrays (real or complex)
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length len(a)+len(b)-1

    Example
    -------
    linear_convolve([1, 2], [3, 4], DFTAnalyzer())
    -> [3., 10., 8.]        # (1+2x)(3+4x) = 3 + 10x + 8x^2
    """
    from transforms import next_power_of_two
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    linear_len = len(a) + len(b) - 1
    N = next_power_of_two(linear_len)

    a_pad = np.zeros(N, dtype=np.complex128); a_pad[:len(a)] = a
    b_pad = np.zeros(N, dtype=np.complex128); b_pad[:len(b)] = b

    C = engine.transform(a_pad) * engine.transform(b_pad)
    c = engine.inverse(C)
    return c[:linear_len].real
```

---

### 3.2 `circular_convolve(a, b, engine)`

```python
def circular_convolve(a: np.ndarray, b: np.ndarray, engine) -> np.ndarray:
    """
    Circular (periodic) convolution of two 1D sequences of equal length N.

    If len(a) != len(b), the shorter one is zero-padded to match.

    Formula:
        (a (*) b)[m] = sum_{n=0}^{N-1}  a[n] * b[(m-n) mod N]

    Via the convolution theorem:
        c = IDFT(DFT(a) * DFT(b))

    Note the difference from linear_convolve: NO extra zero-padding is added.
    The DFT naturally gives circular convolution; Task B's circular mode uses
    this property with the kernel wrapped via modular indexing.

    Parameters
    ----------
    a, b   : 1D numpy arrays
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length N = max(len(a), len(b))

    Example
    -------
    circular_convolve([1, 2, 0, 0], [3, 4, 0, 0], DFTAnalyzer())
    -> [3., 10., 8., 0.]   # same as linear here because we had room

    circular_convolve([1, 2], [3, 4], DFTAnalyzer())
    -> [11., 10.]           # wrap-around: 8 folds into [0] giving 3+8=11
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    N = max(len(a), len(b))
    a_pad = np.zeros(N, dtype=np.complex128); a_pad[:len(a)] = a
    b_pad = np.zeros(N, dtype=np.complex128); b_pad[:len(b)] = b

    C = engine.transform(a_pad) * engine.transform(b_pad)
    return engine.inverse(C).real
```

> **Task B connection:** `convolve_plane(..., circular=True)` relies on this
> property. The wraparound artefact in `wraparound.png` is circular
> convolution wrap showing up spatially.

---

### 3.3 `overlap_add_convolve(x, h, block_size, engine)`

```python
def overlap_add_convolve(x: np.ndarray, h: np.ndarray,
                         block_size: int, engine) -> np.ndarray:
    """
    Overlap-add method for long signal convolution with a short filter.

    Motivation: for a very long signal x (e.g. a megapixel image row) and a
    short filter h (e.g. a 19-tap bokeh kernel), loading the whole signal
    at once wastes memory. The overlap-add method splits x into non-overlapping
    blocks, convolves each block with h (producing tails of length len(h)-1),
    then adds the overlapping tails.

    Steps for each block b of length L:
      1. linear_convolve(b, h)  -> result of length L + len(h) - 1
      2. Accumulate into output[start : start + L + M - 1]  (M = len(h)-1)

    Parameters
    ----------
    x          : 1D signal array (real, arbitrary length)
    h          : 1D filter/kernel array (real, length M)
    block_size : int -- length L of each processing block (choose as power of 2)
    engine     : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length len(x) + len(h) - 1
        Same result as linear_convolve(x, h, engine).

    Complexity
    ----------
    O((len(x)/L) * (L+M) * log(L+M))
    Useful when M << L and we want streaming / memory-bounded processing.
    """
    N_x = len(x)
    M = len(h)
    output_len = N_x + M - 1
    out = np.zeros(output_len, dtype=np.float64)

    pos = 0
    while pos < N_x:
        block = x[pos: pos + block_size]
        result = linear_convolve(block, h, engine)
        end = pos + len(result)
        out[pos:end] += result
        pos += block_size

    return out
```

---

## 4. Signal Reconstruction

### 4.1 `reconstruct_from_magnitude(X, engine)`

```python
def reconstruct_from_magnitude(X: np.ndarray, engine) -> np.ndarray:
    """
    Reconstruct a signal using only the magnitude of its spectrum,
    discarding all phase information (set phase = 0).

    Demonstrates why phase matters: the reconstructed signal looks completely
    different from the original despite having the same power spectrum.

    This connects to Task B's magnitude/phase swap experiment
    (magnitude_phase_swap_template.py).

    Parameters
    ----------
    X      : complex spectrum (DFT output), length N
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64 -- reconstructed signal (zero-phase version)
    """
    magnitude = np.abs(X)
    X_zero_phase = magnitude.astype(np.complex128)   # phase = 0 everywhere
    return engine.inverse(X_zero_phase).real
```

---

### 4.2 `reconstruct_from_phase(X, engine)`

```python
def reconstruct_from_phase(X: np.ndarray, engine) -> np.ndarray:
    """
    Reconstruct a signal using only the phase of its spectrum,
    discarding all magnitude information (set |X[k]| = 1 for all k).

    Together with reconstruct_from_magnitude, this is the basis of the hybrid
    image experiment: swapping magnitudes between two images swaps their
    'content type', while swapping phases swaps their 'structure'.

    Parameters
    ----------
    X      : complex spectrum (DFT output), length N
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64 -- reconstructed signal (unit-magnitude version)
    """
    phase = np.angle(X)
    X_unit_mag = np.exp(1j * phase)   # |X[k]| = 1, phase preserved
    return engine.inverse(X_unit_mag).real
```

---

### 4.3 `low_pass_filter(x, cutoff_bin, engine)`

```python
def low_pass_filter(x: np.ndarray, cutoff_bin: int, engine) -> np.ndarray:
    """
    Ideal rectangular low-pass filter in the frequency domain.

    Sets all bins with |k| > cutoff_bin to zero, then inverse-transforms.
    This is the spectral-domain equivalent of a 'sinc' impulse response in
    the time domain.

    Warning: sharp spectral truncation causes Gibbs ringing in the output
    (see ASSIGNMENT_NOTES.md Section 7.3 and the Windowing Functions section).

    Parameters
    ----------
    x          : 1D real signal, arbitrary length
    cutoff_bin : int -- keep bins 0..cutoff_bin and their conjugate mirrors
    engine     : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, same length as x
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    X = engine.transform(x)
    mask = np.zeros(N, dtype=np.float64)
    mask[:cutoff_bin + 1] = 1.0
    if cutoff_bin > 0:
        mask[N - cutoff_bin:] = 1.0    # mirror (negative frequencies)
    X_filtered = X * mask
    return engine.inverse(X_filtered).real
```

---

## 5. Spectral Properties

### 5.1 `verify_parseval(x, engine, tol=1e-9)`

```python
def verify_parseval(x: np.ndarray, engine, tol: float = 1e-9) -> bool:
    """
    Check Parseval's theorem:  sum|x[n]|^2 = (1/N) * sum|X[k]|^2

    Parseval guarantees that the DFT/FFT is energy-preserving (unitary up to
    the 1/N factor). Failure to satisfy it within floating-point tolerance
    indicates a bug in the transform -- useful as a quick sanity check after
    implementing a new engine.

    Parameters
    ----------
    x      : 1D array (real or complex)
    engine : DFTAnalyzer or FFTTransformer
    tol    : absolute tolerance (default 1e-9)

    Returns
    -------
    bool -- True if Parseval holds within tol

    Example
    -------
    x = np.random.randn(64)
    verify_parseval(x, FFTTransformer())   -> True
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    X = engine.transform(x)
    time_energy = np.sum(np.abs(x) ** 2)
    freq_energy = np.sum(np.abs(X) ** 2) / N
    return abs(time_energy - freq_energy) < tol
```

---

### 5.2 `verify_linearity(x, y, alpha, beta, engine, tol=1e-9)`

```python
def verify_linearity(x, y, alpha, beta, engine, tol=1e-9) -> bool:
    """
    Check linearity:  DFT{a*x + b*y} == a*DFT{x} + b*DFT{y}

    The DFT is a linear operator -- it satisfies superposition exactly.
    This test confirms that the implementation does not introduce nonlinear
    artefacts (e.g. from incorrect rounding or clamping).

    Parameters
    ----------
    x, y        : 1D arrays of equal length N (real or complex)
    alpha, beta : scalar coefficients
    engine      : DFTAnalyzer or FFTTransformer
    tol         : float absolute tolerance

    Returns
    -------
    bool
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    lhs = engine.transform(alpha * x + beta * y)
    rhs = alpha * engine.transform(x) + beta * engine.transform(y)
    return np.max(np.abs(lhs - rhs)) < tol
```

---

### 5.3 `verify_time_shift(x, m, engine, tol=1e-9)`

```python
def verify_time_shift(x: np.ndarray, m: int, engine, tol: float = 1e-9) -> bool:
    """
    Check the time-shift property:
        DFT{x[n - m]} = X[k] * exp(-2*pi*j*k*m/N)

    Shifting the input signal by m samples multiplies every DFT bin k by the
    complex exponential exp(-2*pi*j*k*m/N) -- a linear phase ramp in frequency.
    This property underlies how digital filters implement pure delays.

    Parameters
    ----------
    x      : 1D array, length N
    m      : int, shift amount (can be negative)
    engine : DFTAnalyzer or FFTTransformer
    tol    : float

    Returns
    -------
    bool
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    x_shifted = np.roll(x, m)   # circular shift by m
    X = engine.transform(x)
    X_shifted = engine.transform(x_shifted)
    k = np.arange(N, dtype=np.float64)
    phase_ramp = np.exp(-2j * np.pi * k * m / N)
    X_expected = X * phase_ramp
    return np.max(np.abs(X_shifted - X_expected)) < tol
```

---

### 5.4 `spectral_energy_ratio(X, band_low, band_high)`

```python
def spectral_energy_ratio(X: np.ndarray, band_low: int, band_high: int) -> float:
    """
    Compute the fraction of total spectral energy in bins [band_low, band_high].

    Useful for measuring how much of an image's detail lies in a given
    spatial-frequency band after 2D FFT, or how much energy a blur kernel
    removes from high frequencies.

    Parameters
    ----------
    X          : 1D DFT output (complex), length N
    band_low   : int -- inclusive lower bin index
    band_high  : int -- inclusive upper bin index

    Returns
    -------
    float in [0, 1] -- fraction of total energy in the band

    Example
    -------
    # Gaussian blur suppresses high frequencies:
    # spectral_energy_ratio on high bins should be much less after blurring.
    """
    total = np.sum(np.abs(X) ** 2)
    if total == 0:
        return 0.0
    band = np.sum(np.abs(X[band_low:band_high + 1]) ** 2)
    return float(band / total)
```

---

## 6. 2D Transform Utilities (Task B)

### 6.1 `fftshift_2d(X2d)`

```python
def fftshift_2d(X2d: np.ndarray) -> np.ndarray:
    """
    Shift the 2D DFT so the DC component is at the centre of the array.

    The output of transform_2d places DC at (0, 0).  For display it is
    standard to centre DC at (H//2, W//2).  This is the 2D manual equivalent
    of numpy.fft.fftshift with axes=(0, 1).

    Parameters
    ----------
    X2d : 2D numpy array, shape (H, W), complex

    Returns
    -------
    numpy.ndarray -- same shape and dtype, DC centred

    Usage (Task B)
    --------------
    F = transform_2d(plane, engine)
    F_vis = fftshift_2d(F)
    magnitude = np.log1p(np.abs(F_vis))   # log scale for display
    """
    return np.roll(
        np.roll(X2d, X2d.shape[0] // 2, axis=0),
        X2d.shape[1] // 2, axis=1
    )
```

---

### 6.2 `gaussian_kernel_2d(size, sigma)`

```python
def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D isotropic Gaussian blur kernel of given size and sigma.

    The Gaussian kernel is the 'ideal' smooth blur: its Fourier transform is
    also a Gaussian (no Gibbs ringing), and it is separable, so it can be
    applied as two independent 1D convolutions for extra efficiency.

    Formula:
        G[i, j] = exp(-(i^2 + j^2) / (2*sigma^2))
        then normalised so that sum(G) = 1.

    Parameters
    ----------
    size  : int -- side length (must be odd so there is a clear centre pixel)
    sigma : float -- standard deviation in pixels

    Returns
    -------
    numpy.ndarray of float64, shape (size, size), sum = 1.0

    Task B connection
    -----------------
    image_conv.py builds kernels like this for --kernel gaussian.
    Convolving with this kernel is equivalent to multiplying the image's
    2D spectrum by a Gaussian envelope -- smoothly attenuating high frequencies.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    y, x = np.meshgrid(coords, coords, indexing='ij')
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()
```

---

### 6.3 `separable_kernel_2d(h_row, h_col)`

```python
def separable_kernel_2d(h_row: np.ndarray, h_col: np.ndarray) -> np.ndarray:
    """
    Form a 2D separable kernel from two 1D filters via outer product.

    A 2D filter is separable if it can be written as the outer product of two
    1D filters:  K[i, j] = h_col[i] * h_row[j].

    Separable convolution is a key optimisation: instead of a 2D convolution
    of cost O(H*W*kh*kw), you can apply two 1D convolutions at cost
    O(H*W*(kh+kw)), matching the row-then-column approach used in transform_2d.

    Parameters
    ----------
    h_row : 1D array, length kw -- horizontal filter
    h_col : 1D array, length kh -- vertical filter

    Returns
    -------
    numpy.ndarray, shape (kh, kw)

    Example
    -------
    # Averaging box filter:
    h = np.ones(5) / 5
    box5x5 = separable_kernel_2d(h, h)  # 5x5 box
    """
    h_row = np.asarray(h_row, dtype=np.float64)
    h_col = np.asarray(h_col, dtype=np.float64)
    return np.outer(h_col, h_row)
```

---

### 6.4 `compute_2d_psd(plane, engine)`

```python
def compute_2d_psd(plane: np.ndarray, engine) -> np.ndarray:
    """
    Compute the 2D Power Spectral Density of an image plane.

    PSD[u, v] = |F[u, v]|^2 / (H * W)

    where F is the 2D DFT of the plane (transform every row then every column,
    exactly as in transform_2d from image_conv.py).

    Applying fftshift_2d to the result before display produces the classic
    centred frequency map: low-spatial-frequency content (smooth regions)
    near the centre, high-frequency content (edges, textures) at the edges.

    Parameters
    ----------
    plane  : 2D numpy array (H, W), real-valued image plane (single channel)
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, shape (H, W)

    Usage
    -----
    psd = compute_2d_psd(gray_image, FFTTransformer())
    import matplotlib.pyplot as plt
    plt.imshow(np.log1p(fftshift_2d(psd)), cmap='inferno')
    """
    H, W = plane.shape
    F = np.empty((H, W), dtype=np.complex128)
    for r in range(H):
        F[r, :] = engine.transform(plane[r, :].astype(np.complex128))
    for c in range(W):
        F[:, c] = engine.transform(F[:, c])
    return (np.abs(F) ** 2) / (H * W)
```

---

## 7. Polynomial / Big-Integer Helpers (Task A)

### 7.1 `poly_multiply(p, q, engine)`

```python
def poly_multiply(p: np.ndarray, q: np.ndarray, engine) -> np.ndarray:
    """
    Multiply two polynomials represented as coefficient arrays (little-endian).

    p = [p0, p1, ..., pn]  represents  p0 + p1*x + ... + pn*x^n
    The product has degree deg(p)+deg(q) and coefficients = linear_convolve(p, q).

    This is exactly what multiply_transform in bigmul.py does to each pair of
    digit-limb arrays, then carry-propagation converts the result to a decimal
    string.

    Parameters
    ----------
    p, q   : 1D arrays of polynomial coefficients (int or float), little-endian
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length len(p)+len(q)-1
        Product polynomial coefficients (NOT carry-reduced).

    Example
    -------
    # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
    poly_multiply([1, 2], [3, 4], DFTAnalyzer())  -> [3., 10., 8.]
    """
    return linear_convolve(p, q, engine)
```

---

### 7.2 `poly_evaluate(coeffs, x_val)`

```python
def poly_evaluate(coeffs: np.ndarray, x_val: float) -> float:
    """
    Evaluate a polynomial at a given point using Horner's method.

    Horner's method:
        p(x) = c0 + x*(c1 + x*(c2 + ... + x*cn) ...)
    requires only n multiplications and n additions for a degree-n polynomial.

    This is useful for verifying that two polynomial representations give the
    same value at a test point -- a cheap spot-check of poly_multiply results.

    Parameters
    ----------
    coeffs : 1D array, little-endian polynomial coefficients
    x_val  : float (or complex) -- point to evaluate at

    Returns
    -------
    float (or complex)

    Example
    -------
    # p(x) = 3 + 10x + 8x^2  evaluated at x=10 should give 903
    poly_evaluate([3., 10., 8.], 10)  -> 903.0
    """
    result = 0.0
    for c in reversed(coeffs):
        result = result * x_val + c
    return result
```

---

### 7.3 `count_significant_limbs(limbs, base)`

```python
def count_significant_limbs(limbs: np.ndarray, base: int) -> int:
    """
    Count the number of significant (non-zero leading) limbs after carry.

    After carry propagation in from_limbs(), trailing zeros in the little-endian
    array are stripped. This function gives the same count without modifying the
    array -- useful for benchmarking or estimating the digit count before
    committing to the full string conversion.

    Parameters
    ----------
    limbs : 1D int64 array (little-endian, already carry-reduced)
    base  : int -- the base (10^BASE_DIGITS, e.g. 10000 for BASE_DIGITS=4)

    Returns
    -------
    int -- number of limbs that are significant

    Note: for unreduced limbs (raw convolution output), run carry propagation first.
    """
    i = len(limbs) - 1
    while i > 0 and limbs[i] == 0:
        i -= 1
    return i + 1
```

---

### 7.4 `limb_max_value(n_limbs, base)`

```python
def limb_max_value(n_limbs: int, base: int) -> int:
    """
    Return the maximum integer representable in n_limbs limbs of the given base.

    max = base^n_limbs - 1

    Useful for determining whether a transform result can overflow double
    precision: the largest convolution coefficient is <= n*(base-1)^2 (for n
    input limbs each of value base-1), and this must be < 2^53 ~= 9e15 to
    remain exactly representable.

    Parameters
    ----------
    n_limbs : int -- number of limbs
    base    : int -- e.g. 10**4 = 10000

    Returns
    -------
    int

    Example
    -------
    # For BASE=10^4, n=3 limbs: max = 10^12 - 1 = 999,999,999,999
    limb_max_value(3, 10**4)  -> 999999999999
    """
    return base ** n_limbs - 1
```

---

## 8. Windowing Functions

> **Theory:** A window function tapers a signal to zero at both ends before
> applying the DFT. This reduces **spectral leakage** — the phenomenon where
> energy from one frequency bin bleeds into adjacent bins due to the DFT's
> implicit periodic-extension assumption.
>
> **For convolution (Tasks A & B), windowing is WRONG** — it would distort the
> kernel. Zero-padding is the correct approach. Windowing is used for *spectral
> analysis* of signals (e.g. estimating a signal's frequency content).

### 8.1 `hann_window(N)`

```python
def hann_window(N: int) -> np.ndarray:
    """
    Hann (raised cosine) window of length N.

    w[n] = 0.5 * (1 - cos(2*pi*n/(N-1))),   n = 0..N-1

    The Hann window has excellent frequency selectivity with a first sidelobe
    at -31.5 dB.  It is a popular general-purpose window for power spectrum
    estimation.

    Parameters
    ----------
    N : int -- window length

    Returns
    -------
    numpy.ndarray of float64, shape (N,), values in [0, 1]
    """
    n = np.arange(N, dtype=np.float64)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
```

---

### 8.2 `hamming_window(N)`

```python
def hamming_window(N: int) -> np.ndarray:
    """
    Hamming window of length N.

    w[n] = 0.54 - 0.46 * cos(2*pi*n/(N-1))

    Slightly different weighting from Hann: the first sidelobe is lower
    (-43 dB) but subsequent sidelobes do not fall off as fast.

    Parameters
    ----------
    N : int

    Returns
    -------
    numpy.ndarray of float64, shape (N,)
    """
    n = np.arange(N, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
```

---

### 8.3 `blackman_window(N)`

```python
def blackman_window(N: int) -> np.ndarray:
    """
    Blackman window of length N.

    w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))

    Higher sidelobe suppression (-58 dB) at the cost of a wider main lobe.
    Good for detecting weak tones near strong ones.

    Parameters
    ----------
    N : int

    Returns
    -------
    numpy.ndarray of float64, shape (N,)
    """
    n = np.arange(N, dtype=np.float64)
    return (0.42
            - 0.5  * np.cos(2 * np.pi * n / (N - 1))
            + 0.08 * np.cos(4 * np.pi * n / (N - 1)))
```

---

### 8.4 `apply_window(x, window_fn)`

```python
def apply_window(x: np.ndarray, window_fn) -> np.ndarray:
    """
    Multiply a signal by a window function before spectral analysis.

    Parameters
    ----------
    x         : 1D real signal, length N
    window_fn : callable -- takes N and returns a window array of length N
                (e.g. hann_window, hamming_window, blackman_window)

    Returns
    -------
    numpy.ndarray of float64, same length as x

    Example
    -------
    x_windowed = apply_window(x, hann_window)
    X = DFTAnalyzer().transform(x_windowed.astype(complex))
    # Now |X[k]| shows much less spectral leakage than without windowing.
    """
    x = np.asarray(x, dtype=np.float64)
    w = window_fn(len(x))
    return x * w
```

---

## 9. Correlation

### 9.1 `cross_correlate(x, y, engine)`

```python
def cross_correlate(x: np.ndarray, y: np.ndarray, engine) -> np.ndarray:
    """
    Cross-correlation of two 1D signals via the frequency domain.

    Cross-correlation:
        (x * y)[m] = sum_n  x[n] * y[n + m]   (measure of similarity at lag m)

    Via the correlation theorem (a consequence of the convolution theorem):
        IDFT(conj(DFT(x)) * DFT(y)) = cross-correlation of x and y

    This is equivalent to convolving x with a time-reversed copy of y.

    Difference from convolution:
        convolution :  DFT(x) * DFT(y)          (multiply)
        cross-corr  :  conj(DFT(x)) * DFT(y)    (conjugate then multiply)

    Parameters
    ----------
    x, y   : 1D numpy arrays of equal length N
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length N -- cross-correlation sequence

    Application
    -----------
    The lag at the peak of cross_correlate(x, y) is the time delay between
    x and y -- used in echo detection, synchronisation, and template matching.
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    assert len(x) == len(y), "x and y must have the same length"
    X = engine.transform(x)
    Y = engine.transform(y)
    return engine.inverse(np.conj(X) * Y).real
```

---

### 9.2 `autocorrelate(x, engine)`

```python
def autocorrelate(x: np.ndarray, engine) -> np.ndarray:
    """
    Autocorrelation of a 1D signal -- the cross-correlation of x with itself.

    Autocorrelation:
        R[m] = sum_n  x[n] * x[n + m]

    The autocorrelation at lag 0 (R[0]) equals the total energy of the signal.
    Periodic signals produce periodic autocorrelations with the same period.

    Via the Wiener-Khinchin theorem:
        R = IDFT(|DFT(x)|^2)
    which is cheaper than computing IDFT(conj(DFT(x)) * DFT(x)) because the
    product |X|^2 is a real, non-negative sequence.

    Parameters
    ----------
    x      : 1D numpy array (real or complex)
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    numpy.ndarray of float64, length N
    """
    x = np.asarray(x, dtype=np.complex128)
    X = engine.transform(x)
    psd = np.abs(X) ** 2           # power spectrum (real, non-negative)
    return engine.inverse(psd.astype(np.complex128)).real
```

---

## 10. Number Theoretic Transform (NTT) — Bonus

> **Theory:** Replace complex exponentials with integer modular arithmetic.
> All operations are exact -- no floating-point rounding.
> The standard prime is **p = 998 244 353 = 119 * 2^23 + 1**, which supports
> transforms of length up to 2^23. The primitive root is **g = 3**.

### 10.1 `mod_pow(base, exp, mod)`

```python
def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    Fast modular exponentiation: compute base^exp mod p in O(log exp) steps.

    Used by the NTT to compute primitive roots of unity modulo p.

    Algorithm: repeated squaring (binary exponentiation).
        result = 1
        while exp > 0:
            if exp is odd: result = result * base mod p
            base = base^2 mod p
            exp >>= 1

    Parameters
    ----------
    base, exp, mod : int (non-negative)

    Returns
    -------
    int -- base^exp mod p

    Example
    -------
    mod_pow(3, (998244353 - 1) // 2, 998244353)  -> 1   (Fermat's little theorem)
    """
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result
```

---

### 10.2 `ntt(x, mod=998244353, g=3, inverse=False)`

```python
def ntt(x: list, mod: int = 998244353, g: int = 3, inverse: bool = False) -> list:
    """
    Number Theoretic Transform (NTT) -- integer FFT modulo a prime.

    Identical butterfly structure to FFTTransformer._fft_core, but:
      - twiddle factors are powers of the primitive root g modulo p
      - all arithmetic is mod p (exact integers, no rounding)

    Works for N = power of two, N <= 2^23 (for p = 998244353).

    Forward NTT:
        X[k] = sum  x[n] * g^(kn)  mod p

    Inverse NTT:
        x[n] = (1/N) * sum  X[k] * g^(-kn)  mod p

    Parameters
    ----------
    x       : list of int -- input values (all in [0, p-1])
    mod     : int -- NTT-friendly prime (default 998244353)
    g       : int -- primitive root of mod (default 3)
    inverse : bool -- if True, compute the inverse NTT

    Returns
    -------
    list of int -- NTT or INTT output, all in [0, mod-1]

    Task A connection
    -----------------
    Replace FFTTransformer with this NTT to get exact integer convolution
    for big-integer multiplication -- eliminates all floating-point concerns
    (the Bonus section of the assignment PDF).
    """
    n = len(x)
    if n == 1:
        return x[:]
    if not is_power_of_two(n):
        raise ValueError("NTT length must be a power of two")

    # Bit-reversal permutation
    j = 0
    a = x[:]
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Butterfly stages
    length = 2
    while length <= n:
        half = length // 2
        w_exp = (mod - 1) // length
        if inverse:
            w_exp = (mod - 1) - w_exp   # use g^(-1) direction
        w_base = mod_pow(g, w_exp, mod)

        for start in range(0, n, length):
            w = 1
            for i in range(half):
                u = a[start + i]
                v = a[start + half + i] * w % mod
                a[start + i] = (u + v) % mod
                a[start + half + i] = (u - v) % mod
                w = w * w_base % mod

        length <<= 1

    if inverse:
        inv_n = mod_pow(n, mod - 2, mod)   # modular inverse of n (Fermat)
        a = [xi * inv_n % mod for xi in a]

    return a
```

---

### 10.3 `ntt_convolve(a, b, mod=998244353)`

```python
def ntt_convolve(a: list, b: list, mod: int = 998244353) -> list:
    """
    Exact integer linear convolution via NTT.

    All coefficients in the result are computed exactly modulo p.
    For big-integer multiplication, choose limbs small enough that each
    product coefficient < p ~= 10^9 (use BASE_DIGITS = 2, BASE = 100).

    Parameters
    ----------
    a, b : lists of non-negative integers (each < mod)
    mod  : NTT-friendly prime

    Returns
    -------
    list of int -- exact convolution coefficients mod p,
                   length len(a)+len(b)-1

    Example
    -------
    # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
    ntt_convolve([1, 2], [3, 4])  -> [3, 10, 8]
    """
    from transforms import next_power_of_two
    result_len = len(a) + len(b) - 1
    N = next_power_of_two(result_len)
    a_pad = a + [0] * (N - len(a))
    b_pad = b + [0] * (N - len(b))

    A = ntt(a_pad, mod)
    B = ntt(b_pad, mod)
    C = [(x * y) % mod for x, y in zip(A, B)]
    c = ntt(C, mod, inverse=True)

    return c[:result_len]
```

---

## Quick Self-Tests

Run this block to verify all major functions against the reference transforms:

```python
import numpy as np
from transforms import DFTAnalyzer, FFTTransformer

d = DFTAnalyzer()
f = FFTTransformer()

# --- power-of-two helpers ---
assert is_power_of_two(8) and not is_power_of_two(7)
assert log2_exact(16) == 4

# --- Parseval ---
x = np.random.randn(64)
assert verify_parseval(x, f)

# --- linearity ---
y = np.random.randn(64)
assert verify_linearity(x, y, 2.5, -1.3j, d)

# --- time-shift ---
assert verify_time_shift(x.astype(complex), 5, f)

# --- linear convolution ---
result = linear_convolve([1, 2], [3, 4], d)
assert np.allclose(result, [3., 10., 8.])

# --- circular convolution wrap-around ---
c_circ = circular_convolve([1, 2], [3, 4], d)
assert np.allclose(c_circ, [11., 10.])   # 8 wraps into bin 0

# --- low-pass filter ---
t = np.linspace(0, 1, 64, endpoint=False)
x_sin = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
x_lp = low_pass_filter(x_sin, cutoff_bin=8, engine=f)
X_lp = f.transform(x_lp.astype(complex))
assert np.abs(X_lp[20]) < 1e-9   # high-frequency component gone

# --- dominant frequency ---
assert abs(dominant_frequency(np.sin(2 * np.pi * 5 * t), fs=64) - 5.0) < 1.0

# --- NTT integer convolution ---
assert ntt_convolve([1, 2], [3, 4]) == [3, 10, 8]

print("All extra-function self-tests passed!")
```

---

## Summary Table

| Function | Category | Key Concept |
|---|---|---|
| `is_power_of_two` | Utility | Bit-trick guard for radix-2 FFT |
| `pad_to_power_of_two` | Utility | Zero-padding prerequisite |
| `log2_exact` | Utility | Exact integer log2 |
| `dft_frequency_bins` | Frequency | Maps bin index to physical frequency |
| `fftshift_1d` | Frequency | Centre DC for visualisation |
| `power_spectrum` | Frequency | One-sided PSD (Parseval-preserving) |
| `dominant_frequency` | Frequency | Peak-frequency finder |
| `linear_convolve` | Convolution | Core of Task A polynomial multiplication |
| `circular_convolve` | Convolution | Core of Task B circular mode |
| `overlap_add_convolve` | Convolution | Memory-efficient streaming convolution |
| `reconstruct_from_magnitude` | Reconstruction | Phase importance demo |
| `reconstruct_from_phase` | Reconstruction | Magnitude importance demo |
| `low_pass_filter` | Reconstruction | Ideal rectangular spectral filter |
| `verify_parseval` | Properties | Energy-preservation test |
| `verify_linearity` | Properties | Superposition test |
| `verify_time_shift` | Properties | Phase-ramp property test |
| `spectral_energy_ratio` | Properties | Band-energy analysis |
| `fftshift_2d` | 2D (Task B) | Centre 2D spectrum |
| `gaussian_kernel_2d` | 2D (Task B) | Smooth blur kernel |
| `separable_kernel_2d` | 2D (Task B) | Outer-product decomposition |
| `compute_2d_psd` | 2D (Task B) | Image spatial-frequency analysis |
| `poly_multiply` | Task A | Wrapper on linear convolution |
| `poly_evaluate` | Task A | Horner evaluation (spot-check) |
| `count_significant_limbs` | Task A | Estimate limb count without conversion |
| `limb_max_value` | Task A | Floating-point safety threshold |
| `hann_window` | Windowing | Leakage reduction (spectral analysis) |
| `hamming_window` | Windowing | Leakage reduction variant |
| `blackman_window` | Windowing | Highest sidelobe suppression |
| `apply_window` | Windowing | Apply any window to a signal |
| `cross_correlate` | Correlation | Similarity / time-delay detection |
| `autocorrelate` | Correlation | Wiener-Khinchin, energy at lag 0 |
| `mod_pow` | NTT (Bonus) | Fast modular exponentiation |
| `ntt` | NTT (Bonus) | Exact integer FFT modulo prime |
| `ntt_convolve` | NTT (Bonus) | Exact integer convolution |
