import numpy as np
import matplotlib.pyplot as plt

DT = 0.05 # sampling interval for the time axis
T_MIN, T_MAX = -np.pi, np.pi # x(t) is defined only on this range

def generate_time_axis(t_min=T_MIN, t_max=T_MAX, dt=DT):
    return np.arange(t_min, t_max + dt / 2, dt)


def base_signal(t):
    x = np.sin(t)
    x[(t < T_MIN) | (t > T_MAX)] = 0
    return x

def interpolate_signal(t, x, query_t):
    # np.interp signature is (x, xp, fp) where xp are the known time coordinates.
    # query_t represents the new time points we want to evaluate the signal at.
    # left=0, right=0 enforces the constraint that values outside the range equal 0.
    return np.interp(query_t, t, x, left=0, right=0)

def transform_signal(t, x, alpha, beta):
    # We want to produce y(t) = x(alpha * t + beta)
    # We calculate the required lookup times, and interpolate the values from x(t)
    query_t = alpha * t + beta
    y = interpolate_signal(t, x, query_t)
    return y

def plot_signals(t, x, y, alpha, beta):
    plt.figure(figsize=(9, 5))
    plt.plot(t, x, label="x(t)", linewidth=2)
    plt.plot(t, y, label=f"y(t) = x({alpha}t + {beta})", linewidth=2, linestyle="--")
    plt.title("Time Scaling and Shifting of a Signal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    t = generate_time_axis()
    x = base_signal(t)

    print("Enter alpha and beta to plot y(t) = x(alpha*t + beta).")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        # Take alpha input and check for quit condition
        alpha_input = input("Enter alpha: ")
        if alpha_input.lower() == 'q':
            break
            
        # Take beta input and check for quit condition
        beta_input = input("Enter beta: ")
        if beta_input.lower() == 'q':
            break

        # Validate numerical conversion
        try:
            alpha = float(alpha_input)
            beta = float(beta_input)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q'.\n")
            continue

        y = transform_signal(t, x, alpha, beta)
        plot_signals(t, x, y, alpha, beta)

    print("Exiting.")


if __name__ == "__main__":
    main()