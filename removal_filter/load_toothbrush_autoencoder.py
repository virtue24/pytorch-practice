import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random, copy
import numpy as np

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    # transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
    # transforms.RandomRotation(10),       # Randomly rotate by up to 10 degrees
    # transforms.ColorJitter(brightness=0.2),  # Randomly change brightness
    transforms.ToTensor()  # Tensor values will be in [0, 1] already
])


# Set the path to your local images
data_dir = './data/images'

# Load your local images
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Define the Autoencoder model

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 -> 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # 4 -> 2
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 2048, 2),  # 2 -> 1 (Final bottleneck layer, kernel size is 2 to avoid issues)
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2),  # 1 -> 2
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 2 -> 4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),   # 4 -> 8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),   # 8 -> 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),    # 16 -> 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),     # 32 -> 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),     # 64 -> 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),      # 128 -> 256
            nn.Tanh()  # Output scaled to [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model and load the saved weights
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder_local.pth', map_location=device))
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


for i in range(10):
    # Select a random image from the test dataset
    idx = random.randint(0, len(dataset) - 1)
    original_image, label = dataset[idx]
    original_image = original_image.to(device)

    # Add a random line to the image
    ##modified_image = add_random_line(original_image)
    modified_image = copy.deepcopy(original_image)
    modified_image = modified_image.to(device)

    # Pass the modified image through the autoencoder
    with torch.no_grad():
        reconstructed_image = model(modified_image.unsqueeze(0))  # Add batch dimension

    # Move images to CPU and detach
    original_image_np = original_image.cpu().squeeze().numpy()
    modified_image_np = modified_image.cpu().squeeze().numpy()
    reconstructed_image_np = reconstructed_image.cpu().squeeze().numpy()


    # Compute the absolute difference between the reconstructed and modified images
    difference = np.abs(reconstructed_image_np - modified_image_np)

    # Normalize the difference to [0,1] for display, if necessary
    if np.max(difference) > 0:  # To avoid division by zero
        difference_normalized = difference / np.max(difference)
    else:
        difference_normalized = difference  # No normalization needed if max difference is 0

    # Alternatively, clip the difference to [0,1] in case of any floating point issues
    difference_normalized = np.clip(difference_normalized, 0, 1)

    # Now you can visualize the difference
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