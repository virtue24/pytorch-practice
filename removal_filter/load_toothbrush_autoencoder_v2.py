import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random, copy
import numpy as np
import torch.nn.functional as F

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the transformations (including normalization to [-1, 1])
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Converts image to a tensor in the [0, 1] range with 3 channels
    transforms.Lambda(lambda x: x[[2, 1, 0], ...]),  # Flip RGB to BGR
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Set the path to your local images
data_dir = './data/images'

# Load your local images
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 512 -> 256
            nn.LeakyReLU(0.05),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 32 -> 16
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 16 -> 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # 8 -> 4
            nn.LeakyReLU(0.05),
            nn.Conv2d(1024, 1024, 4),  # 4 -> 1 (Bottleneck)
            nn.LeakyReLU(0.05)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, stride=2),  # 1 -> 4
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 4 -> 8
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 128 -> 256
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 256 -> 512
            nn.Tanh()  # Output scaled to [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# Initialize the model and load the saved weights
model = Autoencoder().to(device)
model.load_state_dict(torch.load('toothbrush_autoencoder_v2.pth', map_location=device))
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

while True:
  # Select a random image from the test dataset
    idx = random.randint(0, len(dataset) - 1)
    original_image, label = dataset[idx]
    original_image = original_image.to(device)

    # Add a random line to the image (as per your custom function, placeholder here)
    modified_image = copy.deepcopy(original_image)
    modified_image = modified_image.to(device)

    with torch.no_grad():
        # Pass the modified image through the autoencoder
        reconstructed_image = model(modified_image.unsqueeze(0))  # Add batch dimension

    # Move the reconstructed image to CPU and detach
    reconstructed_image = reconstructed_image.cpu().squeeze()

    # Move images to CPU and detach
    original_image_np = original_image.cpu().numpy()
    modified_image_np = modified_image.cpu().squeeze().numpy()
    reconstructed_image_np = reconstructed_image.numpy()    

    # Convert from [-1, 1] to [0, 1] for display
    original_image_np = (original_image_np + 1) / 2
    modified_image_np = (modified_image_np + 1) / 2
    reconstructed_image_np = (reconstructed_image_np + 1) / 2

    # If images are in BGR format, convert them to RGB for display
    if original_image_np.shape[0] == 3:  # Assuming 3 channels (BGR format)
        original_image_np = np.transpose(original_image_np, (1, 2, 0))[:, :, ::-1]  # BGR to RGB
        modified_image_np = np.transpose(modified_image_np, (1, 2, 0))[:, :, ::-1]  # BGR to RGB
        reconstructed_image_np = np.transpose(reconstructed_image_np, (1, 2, 0))[:, :, ::-1]  # BGR to RGB

    # Compute the absolute difference between the reconstructed and modified images
    difference = np.abs(reconstructed_image_np - modified_image_np)

    # # Normalize the difference to [0, 1] for display, if necessary
    # if np.max(difference) > 0:  # To avoid division by zero
    #     difference_normalized = difference / np.max(difference)
    # else:
    #     difference_normalized = difference  # No normalization needed if max difference is 0

    # Clip the difference to [0, 1] in case of any floating point issues
    difference_normalized = difference    
    difference_normalized = np.clip(difference_normalized, 0, 1)

    # Convert the difference to a torch tensor for the mean filter application
    difference_tensor = torch.tensor(difference_normalized).permute(2, 0, 1).unsqueeze(0)  # Change shape to [1, C, H, W]

    # Apply a mean filter using torch.nn.functional.avg_pool2d on the difference image
    mean_filtered_difference = F.avg_pool2d(difference_tensor, kernel_size=5, stride=1, padding=1)  # Keep same size with padding

    # Convert the mean filtered difference to single channel by averaging across the RGB channels (if it is RGB)
    if mean_filtered_difference.shape[1] == 3:  # Check if the image has 3 channels
        # Average across the 3 channels to convert to grayscale
        mean_filtered_difference = mean_filtered_difference.mean(dim=1, keepdim=True)  # Average along channel dimension

    # Convert the mean filtered difference back to numpy for visualization
    mean_filtered_difference_np = mean_filtered_difference.squeeze().numpy()

    # Ensure the values are within [0, 1] range (if necessary)
    mean_filtered_difference_np = np.clip(mean_filtered_difference_np, 0, 1)

    # Calculate the mean of the grayscale image
    mean_value = np.mean(mean_filtered_difference_np)

    # Subtract the mean value from each pixel
    adjusted_image = mean_filtered_difference_np - mean_value

    # Apply the conditions:
    # - If the pixel value is less than 0, set it to 0
    # - If the pixel value is between 0 and 1, set it to 1
    adjusted_image = np.where(adjusted_image < 0, 0, adjusted_image)  # Set values below 0 to 0
    adjusted_image = np.where((adjusted_image > 0.3) & (adjusted_image <= 1), 1, adjusted_image)  # Set values between 0 and 1 to 1


    # Now you can visualize the results
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 6, 1)
    plt.title('Original Image')
    plt.imshow(original_image_np)
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title('Modified Image')
    plt.imshow(modified_image_np)
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image_np)
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.title('Difference Image')
    plt.imshow(difference_normalized, cmap='gray')  # Grayscale for difference image
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title('Mean Filtered Difference')
    plt.imshow(mean_filtered_difference_np, cmap='gray')  # Grayscale for mean filtered difference image
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title('Adjusted Image')
    plt.imshow(adjusted_image, cmap='gray')  # Grayscale for adjusted image

    plt.show()