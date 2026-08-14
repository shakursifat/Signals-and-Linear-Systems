"""
CSE220 Online 2 (Step Response)

Instructions:
- Copy (or import) your completed Signal and LTI_System classes from Offline 1.
- Implement the TODO functions below.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt

from signal_lti import DiscreteSignal, LTISystem


def read_signal_from_file(filename: str, INF: int) -> DiscreteSignal:
    with open(filename, "r", encoding="utf-8") as f:
        nstart, nend = map(int, f.readline().strip().split())
        vals = list(map(float, f.readline().strip().split()))
    assert len(vals) == (nend - nstart + 1)
    sig = DiscreteSignal(nstart, nend)
    for i, v in enumerate(vals):
        sig.set_value_at_time(nstart + i, v)
    return sig


def first_difference(sig: DiscreteSignal) -> DiscreteSignal:
    """
    Returns Δsig[n] = sig[n] - sig[n-1] (assume outside range is 0).
    Must use Signal.shift/add/multiply.
    """
    # Delay the signal by 1 to get sig[n-1] and subtract it
    return sig.add(sig.shift(1).multiply(-1))


def impulse_from_step_response(step_response: DiscreteSignal) -> DiscreteSignal:
    """
    Given s[n], compute h[n] = s[n] - s[n-1] (with s[-1]=0).
    Must use only Signal operations.
    """
    return first_difference(step_response)


def output_using_step_response(x: DiscreteSignal, step_response: DiscreteSignal) -> DiscreteSignal:
    """
    Compute y[n] using ONLY step response:
        y = (Δx * s)
    You must reuse your Offline 1 LTI_System machinery (linear combination of impulses).
    """
    # Calculate Δx[n]
    dx = first_difference(x)
    
    # Convolve Δx with the step response s
    system = LTISystem(step_response)
    return system.output(dx)


# Main (demo workflow)
if __name__ == "__main__":
    # Choose INF large enough for your signals
    INF = 50

    # ---- Load provided files ----
    s = read_signal_from_file(r"Onlines_2022\Online 2 _ Convolution\A\Conv_Online_A1_A2\step_response.txt", INF)
    x = read_signal_from_file(r"Onlines_2022\Online 2 _ Convolution\A\Conv_Online_A1_A2\input_signal.txt", INF)

    # ---- Part 1: recover impulse response ----
    h = impulse_from_step_response(s)

    s.plot("Step Response s[n]")
    h.plot("Recovered Impulse Response h[n] = s[n] - s[n-1]")

    # ---- Part 2: output using only step response ----
    dx = first_difference(x)
    y_s = output_using_step_response(x, s)

    x.plot("Input x[n]")
    dx.plot("First Difference Δx[n]")
    y_s.plot("Output y_s[n] computed via step response")

    # ---- Part 3: verify with impulse-response method ----
    sys_h = LTISystem(h)
    y_h = sys_h.output(x)
    y_h.plot("Output y_h[n] computed via impulse response")

    # Check if outputs match closely
    if np.allclose(y_s.values, y_h.values, atol=1e-6):
        print("Outputs match closely!")
    else:
        print("Outputs differ!")

    plt.show()