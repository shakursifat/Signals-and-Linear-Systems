"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    if n <= 1:
        return 1
    # Bit-twiddling trick: fill all lower bits then add 1
    result = 1
    while result < n:
        result <<= 1
    return result


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        # Build the DFT matrix W where W[k,n] = exp(-2j*pi*k*n/N)
        # Using outer product: k is column vector, n is row vector
        k = np.arange(N, dtype=np.float64)
        n = np.arange(N, dtype=np.float64)
        # W[k,n] = exp(-2j*pi*k*n/N)
        W = np.exp(-2j * np.pi * np.outer(k, n) / N)
        return W @ x

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        # Synthesis: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)
        k = np.arange(N, dtype=np.float64)
        n = np.arange(N, dtype=np.float64)
        # W_inv[n,k] = exp(+2j*pi*k*n/N)
        W_inv = np.exp(2j * np.pi * np.outer(n, k) / N)
        return (W_inv @ spectrum) / N


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def _fft_core(self, x):
        """
        Iterative Cooley-Tukey radix-2 DIT FFT.
        x must have length that is a power of two.
        Returns the FFT of x (forward, no normalization).
        """
        N = len(x)
        if N == 1:
            return x.copy()

        # Bit-reversal permutation using bit manipulation
        log2N = int(round(np.log2(N)))
        rev = np.arange(N, dtype=np.int64)
        # Reverse bits
        rev_bits = np.zeros(N, dtype=np.int64)
        tmp = rev.copy()
        for _ in range(log2N):
            rev_bits = (rev_bits << 1) | (tmp & 1)
            tmp >>= 1
        x = x[rev_bits].copy()

        # Butterfly stages - use vectorized NumPy operations per stage
        length = 2
        while length <= N:
            half = length // 2
            # Twiddle factors for this stage: w[m] = exp(-2j*pi*m/length), m=0..half-1
            m = np.arange(half, dtype=np.float64)
            twiddle = np.exp(-2j * np.pi * m / length)

            # Reshape x so that each row is one group of 'length' elements
            # x_shaped[group, pos] where pos in [0, length)
            x_shaped = x.reshape(-1, length)
            # Upper half of each group
            u = x_shaped[:, :half].copy()
            # Lower half multiplied by twiddle
            v = x_shaped[:, half:] * twiddle[np.newaxis, :]
            x_shaped[:, :half] = u + v
            x_shaped[:, half:] = u - v
            # x is a view of x_shaped, so it's updated in place

            length <<= 1

        return x

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        if N == 0:
            return x.copy()
        # Check power of two
        if N & (N - 1) != 0:
            raise ValueError(
                "FFTTransformer.transform: length %d is not a power of two" % N
            )
        return self._fft_core(x.copy())

    def inverse(self, spectrum):
        """
        Inverse FFT, including the 1/N factor.

        Uses the conjugate-transform-conjugate trick to reuse _fft_core:
            IFFT(X) = conj(FFT(conj(X))) / N
        """
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        if N == 0:
            return spectrum.copy()
        if N & (N - 1) != 0:
            raise ValueError(
                "FFTTransformer.inverse: length %d is not a power of two" % N
            )
        # IFFT via conjugate trick
        x = self._fft_core(np.conj(spectrum))
        return np.conj(x) / N


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def _bluestein_fft(self, x):
        """
        Bluestein's chirp-z algorithm for arbitrary N.
        Forward DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)

        Uses the identity:  k*n = -(k-n)^2/2 + k^2/2 + n^2/2
        to rewrite the DFT as a convolution.
        """
        N = len(x)
        if N <= 1:
            return x.copy()

        # If N is already a power of two, just use the radix-2 FFT
        if N & (N - 1) == 0:
            return self._fft_core(x.copy())

        # Chirp sequence: w[n] = exp(-j*pi*n^2/N)
        n = np.arange(N, dtype=np.float64)
        w = np.exp(-1j * np.pi * n * n / N)  # chirp factors

        # a[n] = x[n] * w[n]
        a = x * w

        # The convolution needs length >= 2N - 1; pad to next power of two
        M = next_power_of_two(2 * N - 1)

        # b[n] = conj(w[n]) for n = 0..N-1, and b[M-n] = conj(w[n]) for n=1..N-1
        b = np.zeros(M, dtype=np.complex128)
        b[:N] = np.conj(w)
        b[M - N + 1:] = np.conj(w[1:][::-1])

        # Convolve a (zero-padded) with b using radix-2 FFT
        A = self._fft_core(np.pad(a, (0, M - N)))
        B = self._fft_core(b)
        conv = np.conj(self._fft_core(np.conj(A * B))) / M  # IFFT via conjugate trick

        # Output: X[k] = w[k] * conv[k]  for k=0..N-1
        return w * conv[:N]

    def transform(self, x):
        """Forward FFT for any length N."""
        x = np.asarray(x, dtype=np.complex128)
        return self._bluestein_fft(x)

    def inverse(self, spectrum):
        """
        Inverse FFT for any length N, including 1/N normalization.
        Uses conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N
        """
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        result = self._bluestein_fft(np.conj(spectrum))
        return np.conj(result) / N
