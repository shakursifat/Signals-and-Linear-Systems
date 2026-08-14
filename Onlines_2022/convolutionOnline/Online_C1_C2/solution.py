import numpy as np
import matplotlib.pyplot as plt

# Import your completed classes from your offline file
from signal_lti import DiscreteSignal, LTISystem

# 1. Define SuperSignal class
class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: DiscreteSignal, coefficient=1.0):
        self.components.append((coefficient, signal))
        
# 2. Define extended LTI class with output_super
class SuperLTISystem(LTISystem):
    def output_super(self, super_signal: SuperSignal) -> DiscreteSignal:
        """Takes a SuperSignal object and returns the corresponding output using superposition."""
        if not super_signal.components:
            return DiscreteSignal(0, 0)
            
        # Initialize the final output result using the first component
        first_coeff, first_signal = super_signal.components[0]
        result = self.output(first_signal).multiply(first_coeff)
        
        # Iteratively calculate and add the outputs for the remaining components
        for coeff, sig in super_signal.components[1:]:
            # T{a * x[n]} = a * T{x[n]}
            component_output = self.output(sig).multiply(coeff)
            result = result.add(component_output)
            
        return result


if __name__ == "__main__":

    # ---- Component signals ----
    # x1 has a value of 1 at t=0
    x1 = DiscreteSignal(0, 0)
    x1.set_value_at_time(0, 1.0)

    # x2 has a value of 1 at t=2
    x2 = DiscreteSignal(2, 2)
    x2.set_value_at_time(2, 1.0)

    # ---- Create SuperSignal: x(n) = 2*x1(n) - x2(n) ----
    x_super = SuperSignal()
    x_super.add(x1, 2.0)
    x_super.add(x2, -1.0)

    # ---- Impulse response ----
    # h has a value of 1 at t=0 and 0.5 at t=1
    h = DiscreteSignal(0, 1)
    h.set_value_at_time(0, 1.0)
    h.set_value_at_time(1, 0.5)

    system = SuperLTISystem(h)

    # ---- Output using superposition ----
    y_super = system.output_super(x_super)
    
    # Plot the results
    x1.plot("Component x1[n]")
    x2.plot("Component x2[n]")
    y_super.plot("Output using superposition y[n]")

    # Display all plots at once
    plt.show()