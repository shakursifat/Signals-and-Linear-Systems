import numpy as np
import matplotlib.pyplot as plt


def base_signal(t):
    x = np.exp(-t) * np.cos(t)
    x[(t < -np.pi) | (t > np.pi)] = 0
    return x


def transform_signal(t, x, alpha):
    
    # TODO: implement tranformation
    
    y = np.where(t, alpha*(-x), 0)
    return y


def main():
    t = np.linspace(-np.pi, np.pi, 1000)
    x = base_signal(t)

    while True:
        
        # TODO: complete the loop
        alpha = input()
        if(alpha == 'q') :
            break
        alpha = float(alpha)
        y = transform_signal(t, x, alpha)

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