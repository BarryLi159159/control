import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Lyapunov derivative function
def V_dot(x1, x2):
    return -4 * x1**4 * x2**2

# Generate grid values for x1 and x2
x1_vals = np.linspace(-10, 10, 100)
x2_vals = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
V_dot_vals = V_dot(X1, X2)

# Plotting the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, V_dot_vals, cmap='viridis')

# Labeling the axes
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel(r'$\dot{V}(x_1, x_2)$') 
ax.set_title('3D plot of $\\dot{V}(x_1, x_2)$')

plt.show()
