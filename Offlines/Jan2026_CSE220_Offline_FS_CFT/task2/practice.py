import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

image = imread(f"Offlines/Jan2026_CSE220_Offline_FS_CFT/task2/pikachu.png", mode='L').astype(float)
image = image /np.max(image)

print(image.shape[0])
x = np.linspace(-1, 1, image.shape[1])
print(len(x))
dx = x[1] - x[0]
print(dx)
print(1/(2*dx))
u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), image.shape[1])
print(u)


y = np.linspace(-1, 1, image.shape[0])
print(y)
print(image.shape[1])

# # 1. Shape (Dimensions)
# print("Shape:", image.shape) 
# # Example output: Shape: (1080, 1920)

# # 2. Data Type
# print("Data Type:", image.dtype) 
# # Output will be: float64 (because you used .astype(float))

# # 3. Number of Dimensions
# print("Dimensions:", image.ndim) 
# # Example output: 2

# # 4. Total Size (Pixel Count)
# print("Total Pixels:", image.size) 
# # Example output: 2073600

# print("Minimum pixel value:", image.min())
# print("Maximum pixel value:", image.max())
# print("Average pixel value:", image.mean())

# # Prints a 5x5 grid of pixels from the top-left corner
# print(image[0:5, 0:5])


# # Prints every pixel value in the top row
# # for pixel_value in image[0]:
# #     print(pixel_value)

# print(np.max(image))

