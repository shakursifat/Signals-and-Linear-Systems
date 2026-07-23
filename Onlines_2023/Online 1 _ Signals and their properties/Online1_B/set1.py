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
    
    # TODO: implement interpolation
    
    return np.interp(t, query_t, x)

def transform_signal(t, x, alpha, beta):
    
    # TODO: implement transformation
    
    #y = alpha * x
    y = np.where((t - beta >= -np.pi) & (t - beta <= np.pi), alpha*x, 0)
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
        
        # TODO: complete the loop
        alpha = input("Enter alpha")
        if alpha == 'q' :
            break
        beta = input("Enter beta")

        alpha = float(alpha)
        beta = float(beta)

        y = transform_signal(t, x, alpha, beta)
        plot_signals(t,x, y, alpha, beta)

    print("Exiting.")


if __name__ == "__main__":
    main()