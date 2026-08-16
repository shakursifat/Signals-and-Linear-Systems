import numpy as np

from svg_utils import load_svg_path
from epicycle_animation import save_outputs


class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        """
        Step 1: Store the sampled signal and set up everything the other
        methods will need.

        Parameters
        ----------
        t : 1D numpy array, shape (M,)
            Uniformly spaced sample times covering ONE FULL PERIOD of the
            signal, as a *closed* interval: t[0] == 0 and t[-1] == T (the
            period). This is exactly what svg_utils.load_svg_path(...)
            returns.
        signal : 1D complex numpy array, shape (M,)
            signal[i] = f(t[i]) = x(t[i]) + 1j * y(t[i]). Periodic, so
            signal[-1] == signal[0].
        n_harmonics : int (call it N)
            The series will use every integer harmonic n with
            -N <= n <= N (i.e. 2N+1 terms in total -- do not forget the
            negative harmonics).

        You must set at least the following attributes, since the rest of
        this class (and the provided plotting/animation code) expects
        them to exist:
            self.t, self.signal, self.N
            self.T      -- the period (a float)
            self.omega  -- the fundamental angular frequency, 2*pi/T
            self.coeffs -- an (initially empty) dict that will map
                           n -> c_n once calculate_all_coefficients() has
                           been called
        """
        self.t = t
        self.signal = signal
        self.N = n_harmonics
        self.T = t[-1] - t[0]
        self.omega = 2 * np.pi / self.T
        self.coeffs = {}

    def calculate_cn(self, n):
        """
        Step 2: Compute a single complex Fourier coefficient c_n using
        numerical integration (np.trapezoid) over the stored samples
        self.t, self.signal.

            c_n = (1/T) * integral_0^T  f(t) * exp(-j*n*omega*t)  dt

        n may be zero, positive, or negative.
        """
        integrand = self.signal * np.exp(-1j * n * self.omega * self.t)
        return (1 / self.T) * np.trapezoid(integrand, self.t)

    def calculate_all_coefficients(self):
        """
        Step 3: Populate self.coeffs with c_n for every harmonic
        n = -N, ..., -1, 0, 1, ..., N by repeatedly calling calculate_cn(n).
        """
        for n in range(-self.N, self.N + 1):
            self.coeffs[n] = self.calculate_cn(n)
            #print(abs(self.coeffs[n])*abs(self.coeffs[n]))
            #print(f"{n} = {self.coeffs[n]}")
        #print(len(self.coeffs))

    def approximate(self, t):
        """
        Step 4: Reconstruct (an approximation of) the signal at time(s) t
        from the coefficients already stored in self.coeffs:

            f_hat(t) = sum_{n=-N}^{N} c_n * exp(j*n*omega*t)

        t may be a single number or a numpy array of times -- your
        implementation must support both, since the provided
        plotting/animation code calls this both ways.
        """
        t_arr = np.asarray(t)
        f_hat = np.zeros_like(t_arr, dtype=complex)
        for n in range(-self.N, self.N + 1):
            f_hat += self.coeffs[n] * np.exp(1j * n * self.omega * t_arr)
        
        # If the input was a scalar, return a scalar
        if np.isscalar(t):
            return f_hat.item()
        return f_hat

    
    def prune_harmonics_by_energy(self, r):
        """
        Task 1: Retain the minimal subset of most energetic harmonics 
        that account for at least a fraction `r` of the total energy.
        """
        # 1. Calculate the energy of each harmonic (|c_n|^2)
        energies = {n: np.abs(coeff)**2 for n, coeff in self.coeffs.items()}
        total_energy = sum(energies.values())
        
        # 2. Sort harmonics by energy in descending order
        sorted_harmonics = sorted(energies.keys(), key=lambda n: energies[n], reverse=True)
        
        accumulated_energy = 0.0
        retained_count = 0
        retained_set = set()
        
        # 3. Accumulate energy until the target ratio is met
        target_energy = r * total_energy
        for n in sorted_harmonics:
            accumulated_energy += energies[n]
            retained_count += 1
            retained_set.add(n)
            if accumulated_energy >= target_energy:
                break
                
        # 4. Zero out the coefficients of the discarded harmonics
        for n in self.coeffs.keys():
            if n not in retained_set:
                self.coeffs[n] = 0.0 + 0.0j
                
        actual_energy_ratio = accumulated_energy / total_energy
        return retained_count, actual_energy_ratio

    def evaluate_reconstruction_error(self):
        """
        Task 2: Compute the Mean Squared Error (MSE) between the 
        ground-truth signal and the approximated signal.
        """
        # 1. Generate the reconstructed signal using current coefficients
        f_hat = self.approximate(self.t)
        
        # 2. Compute the MSE
        error_magnitude_squared = np.abs(self.signal - f_hat)**2
        mse = np.mean(error_magnitude_squared)
        
        return mse


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from epicycle_animation import plot_comparison
    from svg_utils import load_svg_path

    # Load the specific heart SVG as instructed
    t, z = load_svg_path("Onlines_2023/Online 3 _ CFT/A/svgs/heart.svg", num_points=1000)
    
    # Define the target energy ratios
    ratios = [0.96, 0.98, 0.99, 1.00]
    
    # Print the formatted table header
    print(f"{'Target Ratio':<15}| {'Harmonics Retained':<20}| {'Actual Energy Ratio':<20}| {'MSE'}")
    print("-" * 75)
    
    for r in ratios:
        # Re-initialize and calculate all coefficients from scratch 
        # to ensure we don't apply pruning to already-pruned data
        fs = FourierEpicycles(t, z, n_harmonics=150)
        fs.calculate_all_coefficients()
        
        # Perform harmonic pruning
        retained_count, actual_ratio = fs.prune_harmonics_by_energy(r)
        
        # Evaluate reconstruction error
        mse = fs.evaluate_reconstruction_error()
        
        # Print the formatted output row
        print(f"{r:<15.2f}| {retained_count:<20}| {actual_ratio:<20.4f}| {mse:.6f}")
        
        # Save the visual comparison plot
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_comparison(fs, z, ax=ax)
        output_filename = f"heart_pruned_{r}.png"
        fig.savefig(output_filename, dpi=120)
        plt.close(fig)
        
        print(f"-> Saved: {output_filename}\n")
