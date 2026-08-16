import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import sys


# =====================================================================
# Given classes — paste your Task 2 implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])


class CFT2D:
    """2D Continuous Fourier Transform. (Given — paste your Task 2 solution)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        Ny, Nx = self.I.shape
        Nu = len(self.u)
        Nv = len(self.v)
        
        Ix_c = np.zeros((Ny, Nu))
        Ix_s = np.zeros((Ny, Nu))
        
        for i in range(Nu):
            cos_term = np.cos(2 * np.pi * self.u[i] * self.x)
            sin_term = np.sin(2 * np.pi * self.u[i] * self.x)
            
            Ix_c[:, i] = np.trapezoid(self.I * cos_term, self.x, axis=1)
            Ix_s[:, i] = np.trapezoid(self.I * sin_term, self.x, axis=1)
            
        real_F = np.zeros((Nv, Nu))
        imag_F = np.zeros((Nv, Nu))
        
        for j in range(Nv):
            cos_term = np.cos(2 * np.pi * self.v[j] * self.y)
            sin_term = np.sin(2 * np.pi * self.v[j] * self.y)
            
            c_c = Ix_c * cos_term[:, np.newaxis]
            s_s = Ix_s * sin_term[:, np.newaxis]
            s_c = Ix_s * cos_term[:, np.newaxis]
            c_s = Ix_c * sin_term[:, np.newaxis]
            
            real_part = c_c - s_s
            imag_part = - (s_c + c_s)
            
            real_F[j, :] = np.trapezoid(real_part, self.y, axis=0)
            imag_F[j, :] = np.trapezoid(imag_part, self.y, axis=0)
            
        return real_F, imag_F


    def plot_magnitude(self):
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        plt.imshow(np.log(1 + magnitude), cmap='magma')
        plt.title('2D CFT Magnitude Spectrum (log-scaled)')
        plt.axis('off')
        plt.show()


class InverseCFT2D:
    """Inverse 2D-CFT. (Given — paste your Task 2 solution)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        Ny = len(self.y)
        Nx = len(self.x)
        Nu = len(self.u)
        Nv = len(self.v)
        
        F_c = np.zeros((Ny, Nu))
        F_s = np.zeros((Ny, Nu))
        
        for j in range(Ny):
            cos_vy = np.cos(2 * np.pi * self.v * self.y[j])
            sin_vy = np.sin(2 * np.pi * self.v * self.y[j])
            
            term_c = self.real * cos_vy[:, np.newaxis] - self.imag * sin_vy[:, np.newaxis]
            term_s = -self.real * sin_vy[:, np.newaxis] - self.imag * cos_vy[:, np.newaxis]
            
            F_c[j, :] = np.trapezoid(term_c, self.v, axis=0)
            F_s[j, :] = np.trapezoid(term_s, self.v, axis=0)
            
        image = np.zeros((Ny, Nx))
        
        for i in range(Nx):
            cos_ux = np.cos(2 * np.pi * self.u * self.x[i])
            sin_ux = np.sin(2 * np.pi * self.u * self.x[i])
            
            term = F_c * cos_ux[np.newaxis, :] + F_s * sin_ux[np.newaxis, :]
            
            image[:, i] = np.trapezoid(term, self.u, axis=1)
            
        return image
# =====================================================================
# Task 1 — band_pass and band_stop filters
# =====================================================================

class FrequencyFilter:

    def high_pass(self, real, imag, cutoff):
        """Given."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

    def band_pass(self, real, imag, r_low, r_high):
        """TODO: retain entries with r_low < d(i,j) <= r_high, zero the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                d_ij = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if d_ij <= r_low or d_ij > r_high :
                    real[i, j] = 0
                    imag[i, j] = 0

        return real, imag


    def band_stop(self, real, imag, r_low, r_high):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                d_ij = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if d_ij > r_low and d_ij <= r_high :
                    real[i, j] = 0
                    imag[i, j] = 0

        return real, imag

    def shift_brightness(self, real, imag, shift_amount):
        """TODO: Task 3. Add shift_amount to the real component of the exact center pixel."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2

        real = real.copy()
        imag = imag.copy()


        real[cx, cy] += shift_amount

        return real, imag

# =====================================================================
# Task 2 — complementarity check on raw spatial reconstructions
# =====================================================================

class ReconstructionValidator:

    def verify_complementarity(self, I_recon, I_bp, I_bs):
        """TODO: verify the complementarity property. Return (is_valid, delta)."""
        rows, cols = I_recon.shape

        max_diff = 0.0
        for i in range(rows):
            for j in range(cols):
                diff = abs(I_bp[i, j] + I_bs[i, j] - I_recon[i, j])
                if (diff > max_diff) :
                    max_diff = diff

        if(max_diff < 1e-9) :
            return True, max_diff
        else :
            return False, max_diff

# =====================================================================
# Entry point (given — do not modify)
# =====================================================================
if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python3 cft_edge_detector.py <input_image>")
    #     sys.exit(1)

    # input_path = sys.argv[1]
    r_low, r_high = 10, 50

    img   = ContinuousImage(f"C:/Users/Sifat/Academics/CSE 220/Offlines/Jan2026_CSE220_Offline_FS_CFT/task2/pikachu_edges.png")
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    filt = FrequencyFilter()
    real_bp, imag_bp = filt.band_pass(real, imag, r_low, r_high)
    real_bs, imag_bs = filt.band_stop(real, imag, r_low, r_high)

    def reconstruct(r, im):
        return InverseCFT2D(r, im, cft2d.u, cft2d.v, img.x, img.y).reconstruct()

    I_recon = reconstruct(real,    imag)
    I_bp    = reconstruct(real_bp, imag_bp)
    I_bs    = reconstruct(real_bs, imag_bs)

    validator = ReconstructionValidator()
    is_valid, delta = validator.verify_complementarity(I_recon, I_bp, I_bs)
    print(f"Complementarity check: {is_valid} | max delta: {delta:.2e}")

    def save_edge_map(I_raw, path):
        edge_map = np.abs(I_raw)
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        plt.imsave(path, 1 - edge_map, cmap='gray')
        print(f"Saved {path}")

    save_edge_map(I_bp, "pikachu_bandpass.png")
    save_edge_map(I_bs, "pikachu_bandstop.png")

    # Task 3 execution
    real_shifted, imag_shifted = filt.shift_brightness(real, imag, shift_amount=2.0)
    I_brightened = reconstruct(real_shifted, imag_shifted)
    
    # Save brightened image (clip to [0,1], no edge-map inversion)
    I_brightened_clipped = np.clip(I_brightened, 0, 1)
    plt.imsave("pikachu_brightened.png", I_brightened_clipped, cmap='gray')
    print("Saved pikachu_brightened.png")
