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
            nn.Conv2d(3, 16, 3, stride=2, padding=1, dilation = 2),  # 512 -> 256
            nn.LeakyReLU(0.05),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, dilation = 2),  # 256 -> 128
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation = 2),  # 128 -> 64
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, dilation = 2),  # 64 -> 32
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, dilation = 2),  # 32 -> 16
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, dilation = 2),  # 16 -> 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(512, 512, 3, stride=2, padding=1, dilation = 2),  # 8 -> 4 (Relaxed Bottleneck)
            nn.LeakyReLU(0.05)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),  # 4 -> 8
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
    # Apply a mean filter to the original image
    original_image_tensor = original_image.unsqueeze(0)  # Add batch dimension

    modified_image = copy.deepcopy(original_image_tensor)
    #modified_image = F.avg_pool2d(original_image_tensor, kernel_size=3, stride=1, padding=1)  # Keep same size with padding
    modified_image = modified_image.to(device)

    with torch.no_grad():
        # Pass the modified image through the autoencoder
        reconstructed_image = model(modified_image)  # Add batch dimension

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

    # Normalize the difference with respect to the maximum value of the corresponding pixel
    # Add a small epsilon to avoid division by zero in case the maximum value is 0
    epsilon = 1e-8
    max_pixel_values = np.maximum(reconstructed_image_np, modified_image_np)
    relative_difference = difference / (max_pixel_values + epsilon)
  
    difference_normalized = difference    
    difference_normalized = np.clip(difference_normalized, 0, 1)

    # Convert the difference to a torch tensor for the mean filter application
    difference_tensor = torch.tensor(difference_normalized).permute(2, 0, 1).unsqueeze(0)  # Change shape to [1, C, H, W]

    # Apply a mean filter using torch.nn.functional.avg_pool2d on the difference image
    mean_filtered_difference = F.avg_pool2d(difference_tensor, kernel_size=3, stride=1, padding=1)  # Keep same size with padding

    # Convert the mean filtered difference to single channel by averaging across the RGB channels (if it is RGB)
    if mean_filtered_difference.shape[1] == 3:  # Check if the image has 3 channels
        # Average across the 3 channels to convert to grayscale
        mean_filtered_difference = mean_filtered_difference.mean(dim=1, keepdim=True)  # Average along channel dimension

    # Convert the mean filtered difference back to numpy for visualization
    mean_filtered_difference_np = mean_filtered_difference.squeeze().numpy()

    # Ensure the values are within [0, 1] range (if necessary)
    mean_filtered_difference_np = np.clip(mean_filtered_difference_np, 0, 1)

    # Calculate the mean of the grayscale image
    # Subtract the mean value from each pixel
    grey_scale_relative_difference = relative_difference.mean(axis=2)  # Calculate mean along the channel dimension
    mean_value = np.mean(grey_scale_relative_difference)
    std_value = np.std(grey_scale_relative_difference)
    adjusted_image = grey_scale_relative_difference - (mean_value+2*std_value) # Subtract the mean value


    # Apply the conditions:
    # - If the pixel value is less than 0, set it to 0
    # - If the pixel value is between 0 and 1, set it to 1
    adjusted_image = np.where(adjusted_image < 0, 0, adjusted_image)  # Set values below 0 to 0
    normalized_adjusted_image = (adjusted_image - np.min(adjusted_image)) / (np.max(adjusted_image) - np.min(adjusted_image))

    filtered_adjusted_image = F.avg_pool2d(torch.tensor(normalized_adjusted_image).unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze().numpy()


    #adjusted_image = np.where((adjusted_image > 0) & (adjusted_image <= 1), 1, adjusted_image)  # Set values between 0 and 1 to 1


    # Now you can visualize the results
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 8, 1)
    plt.title('Original Image')
    plt.imshow(original_image_np)
    plt.axis('off')

    plt.subplot(1, 8, 2)
    plt.title('Modified Image')
    plt.imshow(modified_image_np)
    plt.axis('off')

    plt.subplot(1, 8, 3)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image_np)
    plt.axis('off')

    plt.subplot(1, 8, 4)
    plt.title('Difference Image')
    plt.imshow(difference_normalized, cmap='gray')  # Grayscale for difference image
    plt.axis('off')

    plt.subplot(1, 8, 5)
    plt.title('Relative Difference')
    plt.imshow(relative_difference, cmap='gray')  # Grayscale for mean filtered difference image
    plt.axis('off')

    plt.subplot(1, 8, 6)
    plt.title('Adjusted Image')
    plt.imshow(adjusted_image, cmap='gray')  # Grayscale for adjusted image
    plt.axis('off')

    plt.subplot(1, 8, 7)
    plt.title('Normalized Adjusted Image')
    plt.imshow(filtered_adjusted_image, cmap='gray')  # Grayscale for adjusted image
    plt.axis('off')


    # Flatten the array to get pixel values
    flattened_values = normalized_adjusted_image.flatten()

    # Create bins with 0.025 resolution
    bins = np.arange(0, 1.010, 0.010)

    # Plot the histogram
    # plt.subplot(1, 7, 7)
    # plt.hist(flattened_values, bins=bins, edgecolor='black', alpha=0.7)
    # plt.title('Frequency of Pixel Values (Resolution 0.025)')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    
    plt.show()