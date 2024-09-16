import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
import copy

# Set the path to your local images
data_dir = './data/images'

# Define the transformations (including normalization to [-1, 1])
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Converts image to a tensor in the [0, 1] range with 3 channels
    transforms.Lambda(lambda x: x[[2, 1, 0], ...]),  # Flip RGB to BGR
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load your local images
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=4,
                                          shuffle=True)

# Check if data loaded properly
dataiter = iter(data_loader)
images, labels = next(dataiter)
print('Image tensor min and max values:', torch.min(images), torch.max(images))
print('Image tensor shape:', images.shape)

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
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 -> 32 (latent space)
            nn.LeakyReLU(0.05),

        )
        
        # Decoder
        self.decoder = nn.Sequential(          
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


# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize the model and move it to the device
model = Autoencoder().to(device)

# Define the loss function (L1 Loss) and optimizer
criterion = nn.MSELoss()
# Using AdamW optimizer (better handling of weight decay)
optimizer = optim.AdamW(model.parameters(),
                        lr=5e-4,
                        weight_decay=1e-5)

# Training loop
num_epochs = 250
outputs = []


def stochastically_cut_out(image=None, max_cut_size=64, cut_prob=0.25):
    # Randomly decide whether to apply the cut-out augmentation
    if np.random.rand() > cut_prob:
        return image
    
    # image is expected to be a tensor of shape (batch_size, channels, height, width)
    _, c, h, w = image.shape  # Get the dimensions of the input image

    # Calculate the mean color for each channel (across the whole image)
    mean_color = image.mean(dim=[2, 3], keepdim=True)  # Mean over height and width

    # Randomly select a position and size for the cut-out area
    cut_h = np.random.randint(0, max_cut_size)  # Random height of the cut-out
    cut_w = np.random.randint(0, max_cut_size)  # Random width of the cut-out
    y = np.random.randint(0, h - cut_h)  # Random starting y position
    x = np.random.randint(0, w - cut_w)  # Random starting x position

    # Set the selected section to the mean color
    image[:, :, y:y+cut_h, x:x+cut_w] = mean_color

    print(f'Cut-out applied at position (y, x): ({y}, {x}), size (h, w): ({cut_h}, {cut_w})')
    return image

for epoch in range(num_epochs):
    for (img, _) in data_loader:
        # Move images to the device
        cutted_img = stochastically_cut_out(image=copy.deepcopy(img), max_cut_size=32, cut_prob=0.5)
        img = img.to(device)
        cutted_img = cutted_img.to(device)

        # Forward pass
        recon = model(cutted_img)
        loss = criterion(recon, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

# Save the trained model
torch.save(model.state_dict(), 'toothbrush_autoencoder_v2.pth')
print('Model saved as autoencoder_local.pth')

# Function to convert BGR to RGB and scale from [-1, 1] to [0, 1] for plotting
def bgr_to_rgb(img):
    img = (img + 1) / 2  # Scale pixel values from [-1, 1] to [0, 1]
    return np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # Change channel order and reshape for plotting

# Plotting original and reconstructed images
for k in range(0, num_epochs, 5):
    plt.figure(figsize=(18, 4))  # Increased figure size for larger images
    
    # Move tensors to CPU before converting to NumPy
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    
    # Plot original images
    for i in range(9):
        if i >= len(imgs):
            break
        plt.subplot(2, 9, i + 1)
        # Convert BGR to RGB and scale to [0, 1]
        plt.imshow(bgr_to_rgb(imgs[i]))
        plt.axis('off')
    
    # Plot reconstructed images
    for i in range(9):
        if i >= len(recon):
            break
        plt.subplot(2, 9, 9 + i + 1)
        # Convert BGR to RGB and scale to [0, 1]
        plt.imshow(bgr_to_rgb(recon[i]))
        plt.axis('off')
    
    # Display the figure
    plt.show()
