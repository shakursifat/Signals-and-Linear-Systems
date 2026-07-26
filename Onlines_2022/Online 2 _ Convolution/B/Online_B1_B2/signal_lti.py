import numpy as np


def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = time_values[::step]

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class DiscreteSignal:
    """Finite discrete-time signal with integer indices."""

    # Create a finite discrete-time signal over the given integer range.
    def __init__(self, start_time, end_time):
        if start_time > end_time:
            raise ValueError("start_time cannot be greater than end_time")
        
        self.start_time = start_time
        self.end_time = end_time
        # Initialize the contiguous array with zeros
        self.values = [0.0] * (self.end_time - self.start_time + 1)

    # Return the number of stored samples in the signal.
    def __len__(self):
        return len(self.values)

    # Return the integer time indices covered by the signal.
    def times(self):
        return range(self.start_time, self.end_time + 1)

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        if self.start_time <= t <= self.end_time:
            return self.values[t - self.start_time]
        return 0.0

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        if self.start_time <= t <= self.end_time:
            self.values[t - self.start_time] = float(value)
        else:
            raise ValueError(f"Time {t} is outside the stored range [{self.start_time}, {self.end_time}]")

    # Return a shifted copy of the signal.
    def shift(self, k):
        # A positive k shifts the signal to the right (delays it)
        # Therefore, both bounds increase by k
        shifted_signal = DiscreteSignal(self.start_time + k, self.end_time + k)
        
        # The values remain in the exact same sequential order
        shifted_signal.values = list(self.values)
        return shifted_signal

    # Return the sum of this signal and another signal.
    def add(self, other):
        # Determine the absolute boundaries required to hold both signals
        new_start = min(self.start_time, other.start_time)
        new_end = max(self.end_time, other.end_time)
        
        result = DiscreteSignal(new_start, new_end)
        
        for t in range(new_start, new_end + 1):
            combined_value = self.get_value_at_time(t) + other.get_value_at_time(t)
            result.set_value_at_time(t, combined_value)
            
        return result

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        result = DiscreteSignal(self.start_time, self.end_time)
        
        for t in self.times():
            scaled_value = self.get_value_at_time(t) * scalar
            result.set_value_at_time(t, scaled_value)
            
        return result

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        # Returns a list of (time, value) tuples where the magnitude exceeds tolerance
        nonzero = []
        for t in self.times():
            val = self.get_value_at_time(t)
            if abs(val) > tolerance:
                nonzero.append((t, val))
        return nonzero

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax

class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Store the impulse response that defines the LTI system.
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        out_start = input_signal.start_time + self.impulse_response.start_time
        out_end = input_signal.end_time + self.impulse_response.end_time
        return out_start, out_end

    # Return all shifted and scaled impulse-response components for the input.
    def get_response_components(self, input_signal):
        components = []
        # Extract only the nonzero samples to avoid unnecessary zero-signals
        for k, x_k in input_signal.nonzero_samples():
            # Create the signal component: x[k] * h[n-k]
            component = self.impulse_response.shift(k).multiply(x_k)
            components.append(component)
        return components

    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal):
        out_start, out_end = self.output_range(input_signal)
        
        # Initialize an empty signal that covers the required output bounds
        # Note: DiscreteSignal from previous step needs to be imported/available
        result = DiscreteSignal(out_start, out_end)
        
        components = self.get_response_components(input_signal)
        for component in components:
            result = result.add(component)
            
        return result

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal, n):
        contributions = []
        for k, x_k in input_signal.nonzero_samples():
            # For each non-zero input at time k, find the impulse response at n - k
            h_n_minus_k = self.impulse_response.get_value_at_time(n - k)
            
            # If h[n-k] is also non-zero, this term contributes to the sum
            if h_n_minus_k != 0:
                contributions.append(x_k * h_n_minus_k)
        return contributions

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        # The output at time n is simply the sum of all contributing product terms
        contributions = self.get_contributions_at_time(input_signal, n)
        return sum(contributions)

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        out_start, out_end = self.output_range(input_signal)
        result = DiscreteSignal(out_start, out_end)
        
        # Evaluate the convolution sum directly at every output index
        for n in range(out_start, out_end + 1):
            y_n = self.output_at_time(input_signal, n)
            result.set_value_at_time(n, y_n)
            
        return result
