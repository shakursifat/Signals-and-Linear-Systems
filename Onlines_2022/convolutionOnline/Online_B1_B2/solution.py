import numpy as np
import matplotlib.pyplot as plt

# Import your completed classes from your offline file
from signal_lti import DiscreteSignal, LTISystem

if __name__ == "__main__":
    
    # 1. Define the input signal x[n]
    x = DiscreteSignal(0, 2)
    x.set_value_at_time(0, 1.0)
    x.set_value_at_time(2, -1.0)
    x.plot("Input x[n]")

    # 2. Define the impulse responses based on the diagram
    h1 = DiscreteSignal(0, 0)
    h1.set_value_at_time(0, 1.0)

    h2 = DiscreteSignal(1, 1)
    h2.set_value_at_time(1, 0.5)

    h3 = DiscreteSignal(0, 1)
    h3.set_value_at_time(0, 1.0)
    h3.set_value_at_time(1, 1.0)

    # 3. Initialize the LTI systems
    sys1 = LTISystem(h1)
    sys2 = LTISystem(h2)
    sys3 = LTISystem(h3)
    
    # ==========================================
    # Task A: Determine output block by block
    # ==========================================
    # Pass x through h1 and h2
    y1 = sys1.output(x)
    y2 = sys2.output(x)
    
    # Sum the outputs of the parallel branches
    y_sum = y1.add(y2)
    
    # Pass the summed signal through h3
    y_final_1 = sys3.output(y_sum)
    y_final_1.plot("Output via block-by-block system")

    # ==========================================
    # Task B: Determine h_combined
    # ==========================================
    # The parallel branches add their impulse responses: (h1 + h2)
    h_sum = h1.add(h2)
    
    # The cascaded branch convolves the result: (h1 + h2) * h3
    # We can compute this by passing h_sum as an "input" into sys3
    h_combined = sys3.output(h_sum)
    h_combined.plot("Combined Impulse Response (h_combined)")
    
    # Create the single equivalent system and apply x
    sys_combined = LTISystem(h_combined)
    y_final_2 = sys_combined.output(x)
    y_final_2.plot("Output via combined impulse response")

    # ==========================================
    # Verification
    # ==========================================
    # Check if both final outputs match exactly
    if np.allclose(y_final_1.values, y_final_2.values):
        print("Outputs are equal: True")
    else:
        print("Outputs are equal: False")

    # Display all plots at once
    plt.show()