import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    # Initialize output array with zeros to handle out-of-bounds values
    y_interp = np.zeros_like(t_query)
    
    # Mask to isolate queries that fall within the original time range
    valid_mask = (t_query >= t_original[0]) & (t_query <= t_original[-1])
    t_valid = t_query[valid_mask]
    
    if len(t_valid) > 0:
        # Find insertion indices (side='right' ensures t_original[i-1] <= v < t_original[i])
        idx = np.searchsorted(t_original, t_valid, side='right')
        print(idx)
        # Safely determine left and right indices without array overflow
        left_idx = np.clip(idx - 1, 0, len(t_original) - 1)
        print(left_idx)
        right_idx = np.clip(idx, 0, len(t_original) - 1)
        print(right_idx)
        
        # Determine if the query time exactly matches an existing sample time
        is_exact = np.isclose(t_valid, t_original[left_idx])
        
        # Apply exact value if it exists, otherwise average the left and right neighbors
        y_interp[valid_mask] = np.where(
            is_exact,
            x_original[left_idx],
            0.5 * (x_original[left_idx] + x_original[right_idx])
        )
        
    return y_interp


def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    # Calculate the scaled time array
    t_scaled = t / k
    
    # Pass the scaled time as the query points to the interpolator
    return interpolate_signal(t, x, t_scaled)


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot base signal
    plt.plot(t, x, label="Base Signal: x(t)", color="blue", linewidth=1.5)
    
    # Plot scaled signal
    plt.plot(t, y, label="Scaled Signal: y(t)", color="red", linestyle="--", linewidth=1.5)
    
    # Formatting
    plt.title(title)
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()