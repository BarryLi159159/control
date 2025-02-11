import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define the system matrices
A = np.array([[0, 1],
              [-2, -2]])
B = np.array([[1],
              [1]])
C = np.array([[2, 3]])
D = np.array([[0]])
u = 1
t = 5

# Compute the matrix exponential e^{At}
e_At = expm(A * t)

# solving (A^(-1) * (e^{At} - I) * B)
I = np.eye(A.shape[0])
x_t = np.dot(np.dot(np.linalg.inv(A), (e_At - I)), B)

# y(t) = C * x(t) + D * u(t)
y_t = np.dot(C, x_t) + np.dot(D, 1)  # u(t) = 1
print(f"y(5) = {y_t[0][0]}")

# Sampling time
T = 1

# Discrete-time state-space matrices
A_d = expm(A * T)
print("A_d (Discrete-time A):")
print(A_d)

# B_d: B_d = (A^{-1} * (e^{A*T} - I)) * B
I = np.eye(A.shape[0])  # Identity matrix of same size as A
B_d = np.dot(np.dot(np.linalg.inv(A), (A_d - I)), B)

print("B_d (Discrete-time B):")
print(B_d)

# e^{At}
def continuous_time_response(A, B, C, t_values):
    y_ct = []
    for t in t_values:
        e_At = expm(A * t)  # Matrix exponential
        x_t = np.dot(np.dot(np.linalg.inv(A), (e_At - np.eye(A.shape[0]))), B)
        y_t = np.dot(C, x_t)
        y_ct.append(y_t[0][0])  # Store scalar value of y(t)
    return y_ct

# Time values for continuous 
t_values = np.linspace(0, 5, 100)

# Get continuous-time response
y_ct = continuous_time_response(A, B, C, t_values)

# Discrete-time response
y_d = []  # To store the output y[k] for discrete-time
x_d = np.array([[0], [0]])  # Initial state 
for k in range(6):  # We want to calculate up to y[5]
    y_k = np.dot(C, x_d) + D * u  # Compute y[k]
    y_d.append(y_k[0][0])  # y[k]
    x_d = np.dot(A_d, x_d) + B_d * u  # Update the state x[k+1]
k_values = np.arange(6)

# Plot 
plt.figure()
plt.plot(t_values, y_ct, label="Continuous-Time y(t)", color='b', linewidth=2)

plt.plot(k_values, y_d, label="Discrete-Time y[k]", color='r', marker='o', linestyle='-', linewidth=2)

plt.xlabel('Time (t) or Time Step (k)')
plt.ylabel('Output y(t) or y[k]')
plt.title('Continuous-Time vs Discrete-Time Response')
plt.legend()
plt.grid(True)
plt.show()