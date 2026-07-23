import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np

# Provided data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
north = np.array([120, 135, 128, 142, 150, 158, 162, 170, 168, 180, 185, 195])
south = np.array([110, 118, 125, 130, 138, 145, 150, 155, 160, 168, 172, 180])
central = np.array([100, 108, 115, 120, 130, 140, 148, 152, 158, 165, 170, 175])

# Figure Settings: Create a figure of size (14, 10)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# -------------------------------------------------------------------------
# Plot 1 (Top Left): Monthly Profit Line Plot
# -------------------------------------------------------------------------
axs[0, 0].plot(months, north, label='North', color='blue', linestyle='-')
axs[0, 0].plot(months, south, label='South', color='green', linestyle='--')
axs[0, 0].plot(months, central, label='Central', color='red', linestyle=':')
axs[0, 0].set_title('Monthly Branch Profit (2025)')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Profit (in thousand dollars)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# -------------------------------------------------------------------------
# Plot 2 (Top Right): December Profit Comparison
# -------------------------------------------------------------------------
branches = ['North', 'South', 'Central']
dec_profits = [north[-1], south[-1], central[-1]]
colors = ['blue', 'green', 'red']

axs[0, 1].bar(branches, dec_profits, color=colors)
axs[0, 1].set_title('December Profit Comparison')
axs[0, 1].set_xlabel('Branch')
axs[0, 1].set_ylabel('Profit (in thousand dollars)')
axs[0, 1].grid(True, axis='y')

# -------------------------------------------------------------------------
# Plot 3 (Bottom Left): North Branch Profit Distribution
# -------------------------------------------------------------------------
axs[1, 0].scatter(months, north, color='blue', s=60)
axs[1, 0].set_title('North Branch Profit Distribution')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Profit (in thousand dollars)')
axs[1, 0].grid(True)

# -------------------------------------------------------------------------
# Plot 4 (Bottom Right): Quarterly Branch Profit
# -------------------------------------------------------------------------
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Computing segment slices for quarters
q_north = [np.sum(north[0:3]), np.sum(north[3:6]), np.sum(north[6:9]), np.sum(north[9:12])]
q_south = [np.sum(south[0:3]), np.sum(south[3:6]), np.sum(south[6:9]), np.sum(south[9:12])]
q_central = [np.sum(central[0:3]), np.sum(central[3:6]), np.sum(central[6:9]), np.sum(central[9:12])]

# Stacked bar setup: bottom layers aggregate existing stacks
axs[1, 1].bar(quarters, q_north, label='North', color='blue')
axs[1, 1].bar(quarters, q_south, bottom=q_north, label='South', color='green')
axs[1, 1].bar(quarters, np.array(q_north) + np.array(q_south), bottom=np.array(q_north) + np.array(q_south), 
              label='Central', color='red')
# Fixing the explicit stack addition tracking for the third bar segment:
axs[1, 1].cla() # Clear incorrect bottom reference setup and redo cleanly
axs[1, 1].bar(quarters, q_north, label='North', color='blue')
axs[1, 1].bar(quarters, q_south, bottom=q_north, label='South', color='green')
axs[1, 1].bar(quarters, q_central, bottom=np.array(q_north) + np.array(q_south), label='Central', color='red')

axs[1, 1].set_title('Quarterly Branch Profit')
axs[1, 1].set_xlabel('Quarter')
axs[1, 1].set_ylabel('Profit (in thousand dollars)')
axs[1, 1].legend()
axs[1, 1].grid(True, axis='y')

# Main overall settings
fig.suptitle('Company Branch Performance Analysis (2025)', fontsize=16)
plt.tight_layout()

# Display configuration
plt.show()