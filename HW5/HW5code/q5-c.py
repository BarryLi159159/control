import numpy as np
import matplotlib.pyplot as plt

# Define the system dynamics for the original system and linearized system
def original_system(x, y):
    x_dot = y - x * y**2
    y_dot = -x**3
    return x_dot, y_dot

def linearized_system(x, y):
    x_dot = y
    y_dot = 0
    return x_dot, y_dot

# Generate a grid of points
x_vals = np.linspace(-2, 2, 20)  
y_vals = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate the vector field for original system
U_orig, V_orig = original_system(X, Y)
U_lin, V_lin = linearized_system(X, Y)

# Create the figure and plot 
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Original System
ax[0].quiver(X, Y, U_orig, V_orig, color='blue')  # Plot vector field with arrows
ax[0].set_title('Phase Portrait: Original System')
ax[0].set_xlabel('$x_1$')
ax[0].set_ylabel('$x_2$')
ax[0].grid(True)

# Linearized System
ax[1].quiver(X, Y, U_lin, V_lin, color='red')  
ax[1].set_title('Phase Portrait: Linearized System')
ax[1].set_xlabel('$x_1$')
ax[1].set_ylabel('$x_2$')
ax[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()
