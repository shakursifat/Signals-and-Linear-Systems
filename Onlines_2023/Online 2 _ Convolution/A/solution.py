import numpy as np
import matplotlib.pyplot as plt

from signal_lti import DiscreteSignal, LTISystem


def make_signal(start_time, end_time, values):
    """Helper: build a DiscreteSignal from a list of values."""
    signal = DiscreteSignal(start_time, end_time)
    for offset, value in enumerate(values):
        signal.set_value_at_time(start_time + offset, value)
    return signal


def max_absolute_difference(first_signal, second_signal):
    """Helper: largest |difference| between two signals over their combined range."""
    min_time = min(first_signal.start_time, second_signal.start_time)
    max_time = max(first_signal.end_time, second_signal.end_time)

    # Use max() with a generator expression for clean, efficient execution
    return max(
        abs(first_signal.get_value_at_time(t) - second_signal.get_value_at_time(t))
        for t in range(min_time, max_time + 1)
    )


# ---- Generic property testers ----
# These must work for ANY apply_system callable
# method such as system_a.output. Do not assume apply_system is an LTISystem.

def test_linearity(apply_system, x1, x2, a, b):
    # Calculate combined input: a*x1 + b*x2
    combined_input = x1.multiply(a).add(x2.multiply(b))
    
    # Calculate output of the combined input: T(a*x1 + b*x2)
    output_of_combined = apply_system(combined_input)
    
    # Calculate scaled combination of individual outputs: a*T(x1) + b*T(x2)
    combined_outputs = apply_system(x1).multiply(a).add(apply_system(x2).multiply(b))
    
    # Return max| apply_system(a*x1 + b*x2)  -  (a*apply_system(x1) + b*apply_system(x2)) |
    return max_absolute_difference(output_of_combined, combined_outputs)


def test_time_invariance(apply_system, x, k):
    # Calculate output of the shifted input: T(x shifted by k)
    output_of_shifted = apply_system(x.shift(k))
    
    # Calculate the shifted version of the standard output: (T(x)) shifted by k
    shifted_output = apply_system(x).shift(k)
    
    # Return max| apply_system(x shifted by k)  -  (apply_system(x) shifted by k) |
    return max_absolute_difference(output_of_shifted, shifted_output)


# ---- System B: y[n] = n * x[n] ----

def system_b(input_signal):
    # Build and return a DiscreteSignal where output[n] = n * input_signal[n]
    result = DiscreteSignal(input_signal.start_time, input_signal.end_time)
    
    for t in input_signal.times():
        val = t * input_signal.get_value_at_time(t)
        result.set_value_at_time(t, val)
        
    return result


def main():
    tolerance = 1e-9

    # ---- Given signals and scalars (do not change) ----
    x1 = make_signal(-2, 2, [1, 0, 2, -1, 3])
    x2 = make_signal(-1, 3, [2, -3, 0, 1, 1])
    a, b = 2.0, -3.0
    k = 3

    h = make_signal(0, 2, [1.0, 0.5, 0.25])
    
    # Initialize System A using LTISystem class
    sys_a = LTISystem(h)

    # Test both properties for system A
    print("=== System A: genuine LTI system (LTISystem.output) ===")
    diff_linear_a = test_linearity(sys_a.output, x1, x2, a, b)
    diff_ti_a = test_time_invariance(sys_a.output, x1, k)
    
    print(f"Linearity max diff:        {diff_linear_a}")
    print(f"Time-invariance max diff:  {diff_ti_a}")

    print()

    # Test both properties for system B
    print("=== System B: y[n] = n * x[n] ===")
    diff_linear_b = test_linearity(system_b, x1, x2, a, b)
    diff_ti_b = test_time_invariance(system_b, x1, k)
    
    print(f"Linearity max diff:        {diff_linear_b}")
    print(f"Time-invariance max diff:  {diff_ti_b}")

    print()
    
    # Print a short conclusion stating which property System B fails
    print("Conclusion: System B fails the time-invariance property, making it a time-varying system.")


if __name__ == "__main__":
    main()