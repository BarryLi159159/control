#Q1
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define the parameters for the system matrices
m = 1888.6  # mass of the vehicle in kg
Iz = 25854  # yaw inertia in kg*m^2
lf = 1.55   # distance from front axle to CG in m
lr = 1.39   # distance from rear axle to CG in m
C_alpha = 20000  # cornering stiffness of each tire in N/rad

# Define a function to construct A and B matrices based on longitudinal speed (v)
def get_system_matrices(v):
    A = np.array([
        [0, 1, 0, 0],
        [0, -4 * C_alpha / (m * v), 4 * C_alpha / m, -2 * C_alpha * (lf - lr) / (m * v)],
        [0, 0, 0, 1],
        [0, -2 * C_alpha * (lf - lr) / (Iz * v), 2 * C_alpha * (lf - lr) / Iz, -2 * C_alpha * (lf**2 + lr**2) / (Iz * v)]
    ])

    B = np.array([
        [0, 0],
        [2 * C_alpha / m, 0],
        [0, 0],
        [2 * C_alpha * lf / Iz, 0]
    ])

    return A, B

# Function to check controllability and observability using P and Q matrices
def check_controllability_observability(A, B, C):
    # Construct controllability matrix P
    P = np.hstack((B, np.dot(A, B), np.dot(np.linalg.matrix_power(A, 2), B), np.dot(np.linalg.matrix_power(A, 3), B)))
    rank_controllability = np.linalg.matrix_rank(P)
    
    # Construct observability matrix Q
    Q = np.vstack((C, np.dot(C, A), np.dot(C, np.linalg.matrix_power(A, 2)), np.dot(C, np.linalg.matrix_power(A, 3))))
    rank_observability = np.linalg.matrix_rank(Q)
    
    return rank_controllability == A.shape[0], rank_observability == A.shape[0]

# Define identity matrix for C (observing all states)
C = np.identity(4)

# Check controllability and observability at the specified velocities
velocities = [2, 5, 8]
results = {}

for v in velocities:
    A, B = get_system_matrices(v)
    is_controllable, is_observable = check_controllability_observability(A, B, C)
    results[v] = {
        "Controllable": is_controllable,
        "Observable": is_observable
    }

# Display results
print("Controllability and Observability Results:")
for v, res in results.items():
    print(f"Velocity {v} m/s -> Controllable: {res['Controllable']}, Observable: {res['Observable']}")

##--------------------------------------------------
# Define the velocity range for analysis
velocity_range = np.arange(1, 41)  # 1 to 40 m/s

# Initialize lists to store results
singular_value_ratios = []
poles_real_parts = []

# Loop over each velocity to compute singular value ratio and poles
for v in velocity_range:
    A, B = get_system_matrices(v)
    
    # Calculate the controllability matrix P
    P = np.hstack((B, np.dot(A, B), np.dot(np.linalg.matrix_power(A, 2), B), np.dot(np.linalg.matrix_power(A, 3), B)))
    
    # Compute singular values and calculate the log ratio
    singular_values = np.linalg.svd(P, compute_uv=False)
    singular_value_ratio = np.log10(singular_values[0] / singular_values[-1])
    singular_value_ratios.append(singular_value_ratio)
    
    # Calculate the poles (eigenvalues of A) and store their real parts
    poles = np.linalg.eigvals(A)
    poles_real_parts.append([np.real(pole) for pole in poles])

# Plot singular value ratio versus velocity
plt.figure(figsize=(10, 6))
plt.plot(velocity_range, singular_value_ratios, label=r'$\log_{10}(\sigma_1 / \sigma_n)$')
plt.xlabel('Velocity (m/s)')
plt.ylabel(r'$\log_{10}(\sigma_1 / \sigma_n)$')
plt.title('Singular Value Ratio vs Velocity')
plt.legend()
plt.grid(True)
plt.show()

# Plot real parts of poles versus velocity with separate subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

for i in range(4):  # Four poles
    axes[i].plot(velocity_range, [poles_real[i] for poles_real in poles_real_parts], label=f'Pole {i+1}')
    axes[i].set_ylabel('Real Part')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

axes[-1].set_xlabel('Velocity (m/s)')
fig.suptitle('Real Part of Poles vs Velocity')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

