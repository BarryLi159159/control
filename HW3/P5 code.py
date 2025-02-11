import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# Load the image (already in grayscale)
gray_image = imread('/Users/li/Desktop/cmu文件/control/HW3/CMU_Grayscale.png', as_gray=True)

# Perform SVD
U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)

# Function to reconstruct the image with a given number of singular values
def reconstruct_image(U, S, Vt, num_singular_values):
    S_truncated = np.zeros((num_singular_values, num_singular_values))
    np.fill_diagonal(S_truncated, S[:num_singular_values])
    U_truncated = U[:, :num_singular_values]
    Vt_truncated = Vt[:num_singular_values, :]
    return np.dot(U_truncated, np.dot(S_truncated, Vt_truncated))

# Interactive function to display image for a given percentage of singular values
def display_compressed_image(percentage):
    # Calculate the number of singular values based on the percentage
    num_singular_values = int(percentage / 100 * total_singular_values)
    
    # Reconstruct the image
    compressed_image = reconstruct_image(U, S, Vt, num_singular_values)
    
    # Plot the reconstructed image
    plt.figure(figsize=(6, 6))
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'{percentage}% Compression ({num_singular_values} singular values)')
    plt.show()

# Calculate total singular values available
m, n = gray_image.shape
total_singular_values = min(m, n)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Show compressed images for 50%, 10%, and 5% compression levels
display_compressed_image(50)  # 50% compression
display_compressed_image(10)  # 10% compression
display_compressed_image(5)   # 5% compression
