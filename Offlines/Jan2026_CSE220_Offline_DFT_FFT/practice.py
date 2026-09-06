import numpy as np
from transforms import *

# x = 5
# y = next_power_of_two(x)

# k = np.arange(5, dtype=np.float64)
# print(k)
# n = np.arange(5, dtype=np.float64)
# print(n)
# print(np.outer(k,n))
# W = np.exp(-2j * np.pi * np.outer(k, n) / 5)

# print(W)

# print(W @ k)

# print(k @ W)

# def fft(x):
#     x = np.asarray(x, dtype=np.complex128)
#     N = len(x)
#     if N == 0 or N & (N - 1):
#         raise ValueError("Input length must be a positive power of two")
#     if N == 1:
#         return x.copy()

#     # Bit-reversal permutation using bit manipulation
#     log2N = int(round(np.log2(N)))
#     print(log2N)
#     rev = np.arange(N, dtype=np.int64)
#     rev_bits = np.zeros(N, dtype=np.int64)
#     print(rev_bits)
#     tmp = rev.copy()
#     print(tmp)
#     print(rev_bits << 1)
#     print(tmp & 1)
#     for _ in range(log2N):
#         rev_bits = (rev_bits << 1) | (tmp & 1)
#         print(rev_bits)
#         tmp >>= 1
#         print(tmp)
#     x = x[rev_bits].copy()

#     # Butterfly stages - use vectorized NumPy operations per stage
#     length = 2
#     while length <= N:
#         half = length // 2
#         # Twiddle factors for this stage: w[m] = exp(-2j*pi*m/length)
#         m = np.arange(half, dtype=np.float64)
#         twiddle = np.exp(-2j * np.pi * m / length)

#         x_shaped = x.reshape(-1, length)
#         u = x_shaped[:, :half].copy()
#         v = x_shaped[:, half:] * twiddle[np.newaxis, :]
#         x_shaped[:, :half] = u + v
#         x_shaped[:, half:] = u - v

#         length <<= 1

#     return x


# print(fft(np.arange(8)))


# BASE_DIGITS = 4
# BASE = 10 ** BASE_DIGITS


# def to_limbs(text, base_digits=BASE_DIGITS):
#     """
#     Convert a decimal string into polynomial coefficients.

#     "123456789" with base_digits = 4 becomes the little-endian limb array
#     [6789, 2345, 1] -- that is, 1*BASE^2 + 2345*BASE^1 + 6789*BASE^0.

#     Parameters
#     ----------
#     text : str
#         A decimal integer, possibly with a leading '+' or '-'.
#     base_digits : int
#         Decimal digits per limb.

#     Returns
#     -------
#     (int, numpy.ndarray)
#         The sign (+1 or -1) and the little-endian limb array (dtype int64).
#         Handle the sign separately from the magnitude: the transform never
#         sees it.
#     """
#     text = text.strip()
#     # Extract sign
#     if text.startswith('-'):
#         sign = -1
#         digits_str = text[1:]
#     elif text.startswith('+'):
#         sign = 1
#         digits_str = text[1:]
#     else:
#         sign = 1
#         digits_str = text

#     # Remove leading zeros but keep at least one digit
#     digits_str = digits_str.lstrip('0') or '0'
#     print(digits_str)

#     # Pack digits into limbs (little-endian: least significant first)
#     # Pad on the left so that length is a multiple of base_digits
#     remainder = len(digits_str) % base_digits
#     if remainder != 0:
#         digits_str = '0' * (base_digits - remainder) + digits_str
#     print(digits_str)

#     # Split into groups of base_digits, then reverse for little-endian
#     base = 10 ** base_digits
#     limbs = []
#     print(range(len(digits_str) - base_digits, -1, -base_digits))
#     for i in range(len(digits_str) - base_digits, -1, -base_digits):
#         print(i)
#         limbs.append(int(digits_str[i:i + base_digits]))

#     return sign, np.array(limbs, dtype=np.int64)


# x, y = to_limbs("+123456789")

# print(x, y)




import argparse
import os

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from image_utils import (load_image, make_kernel, save_comparison, save_image,
                         save_kernel_preview)
from io_utils import write_report
from transforms import DFTAnalyzer, FFTTransformer, next_power_of_two


kernel1 = make_kernel("bokeh", radius=9)

print(kernel1.shape)
print(kernel1)


kernel2 = make_kernel("box", radius=1)

print(kernel2.shape)
print(kernel2)


kernel3 = make_kernel("gaussian", radius=1)

print(kernel2.shape)
print(kernel2)