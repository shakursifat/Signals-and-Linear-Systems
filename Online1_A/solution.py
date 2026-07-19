import numpy as np
import matplotlib.pyplot as plt


def base_signal(t):
    x = np.exp(-t) * np.cos(t)
    x[(t < -np.pi) | (t > np.pi)] = 0
    return x


def transform_signal(t, x, alpha):
    # Time reversal: x(-t) is achieved by reversing the array (since t is symmetric)
    # Amplitude scaling: multiply the reversed signal by alpha
    y = alpha * x[::-1]
    return y


def main():
    t = np.linspace(-np.pi, np.pi, 1000)
    x = base_signal(t)

    # Make an infinite loop that will take alpha repeatedly[cite: 2]
    while True:
        alpha_input = input("Enter alpha (or 'q' to quit): ")
        
        # The program will end upon receiving 'q'[cite: 2]
        if alpha_input.lower() == 'q':
            break
            
        try:
            alpha = float(alpha_input)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
            continue
            
        y = transform_signal(t, x, alpha)

        # Plot (with proper labels and legend) on the same figure: x(t), y(t)[cite: 2]
        plt.figure(figsize=(8, 5))
        plt.plot(t, x, label='x(t)')
        plt.plot(t, y, label=f'y(t) = {alpha} * x(-t)')
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.title('Time Reversal and Amplitude Scaling of x(t)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()