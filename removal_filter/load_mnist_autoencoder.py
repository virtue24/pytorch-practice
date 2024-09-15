import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import numpy as np

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the transformations (same as during training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST test dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load the saved autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: N x 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: N x 32 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # Output: N x 64 x 1 x 1
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # Output: N x 32 x 7 x 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: N x 16 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Output: N x 1 x 28 x 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model and load the saved weights
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder_mnist.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Function to add a random line to an image tensor
def add_random_line(image_tensor):
    image_np = image_tensor.squeeze().cpu().numpy()  # Convert tensor to numpy array
    image_np = (image_np * 0.5) + 0.5  # Unnormalize the image to [0,1]

    # Get random parameters for the line
    x1 = random.randint(0, 27)
    y1 = random.randint(0, 27)
    x2 = random.randint(0, 27)
    y2 = random.randint(0, 27)
    thickness = random.randint(1, 2)
    color = random.uniform(0.99, 1)  # Line color (grayscale value between 0 and 1)

    # Draw the line on the image
    import cv2  # OpenCV library for drawing functions
    image_np = np.uint8(image_np * 255)  # Convert to uint8
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)  # Convert to BGR for cv2.line

    # Draw the line
    cv2.line(image_np, (x1, y1), (x2, y2),
             (int(color * 255), int(color * 255), int(color * 255)),
             thickness)

    # Convert back to grayscale
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    image_np = image_np / 255.0  # Normalize to [0,1]

    # Re-normalize to [-1,1]
    image_np = (image_np - 0.5) / 0.5

    # Ensure the numpy array is of type float32
    image_np = image_np.astype(np.float32)

    # Convert back to tensor
    modified_image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    return modified_image_tensor

# Select a random image from the test dataset
idx = random.randint(0, len(mnist_test) - 1)
original_image, label = mnist_test[idx]
original_image = original_image.to(device)

# Add a random line to the image
modified_image = add_random_line(original_image)
modified_image = modified_image.to(device)

# Pass the modified image through the autoencoder
with torch.no_grad():
    reconstructed_image = model(modified_image.unsqueeze(0))  # Add batch dimension

# Move images to CPU and detach
original_image_np = original_image.cpu().squeeze().numpy()
modified_image_np = modified_image.cpu().squeeze().numpy()
reconstructed_image_np = reconstructed_image.cpu().squeeze().numpy()

# Unnormalize the images for display
original_image_np = (original_image_np * 0.5) + 0.5  # From [-1,1] to [0,1]
modified_image_np = (modified_image_np * 0.5) + 0.5  # From [-1,1] to [0,1]
# reconstructed_image_np is already in [0,1], no need to unnormalize

# Compute the absolute difference
difference = np.abs(reconstructed_image_np - modified_image_np)

# Normalize the difference to [0,1] for display
# Since the difference can range from 0 to a maximum value, we normalize it by dividing by the maximum value
difference_normalized = difference / np.max(difference)

# Alternatively, you can clip the difference to [0,1] in case of any floating point issues
difference_normalized = np.clip(difference_normalized, 0, 1)

# Plot the images
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(original_image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Modified Image')
plt.imshow(modified_image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Difference Image')
plt.imshow(difference_normalized, cmap='gray')
plt.axis('off')

plt.show()
