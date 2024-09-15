import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set the path to your local images
data_dir = './data/images'

# Define the transformations (with augmentation and [0, 1] normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    # transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
    # transforms.RandomRotation(10),       # Randomly rotate by up to 10 degrees
    # transforms.ColorJitter(brightness=0.2),  # Randomly change brightness
    transforms.ToTensor()  # Tensor values will be in [0, 1] already
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
# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize the model and move it to the device
model = Autoencoder().to(device)

# Define the loss function (L1 Loss) and optimizer
criterion = nn.L1Loss()
# Using AdamW optimizer (better handling of weight decay)
optimizer = optim.AdamW(model.parameters(),
                        lr=5e-4,
                        weight_decay=1e-5)

# Training loop
num_epochs = 100
outputs = []

for epoch in range(num_epochs):
    for (img, _) in data_loader:
        # Move images to the device
        img = img.to(device)

        # Forward pass
        recon = model(img)
        loss = criterion(recon, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

# Save the trained model
torch.save(model.state_dict(), 'autoencoder_local.pth')
print('Model saved as autoencoder_local.pth')

# Plotting original and reconstructed images
for k in range(0, num_epochs, 5):
    plt.figure(figsize=(18, 4))  # Increased figure size for larger images
    plt.gray()
    # Move tensors to CPU before converting to NumPy
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    # Plot original images
    for i in range(9):
        if i >= len(imgs):
            break
        plt.subplot(2, 9, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.axis('off')
    # Plot reconstructed images
    for i in range(9):
        if i >= len(recon):
            break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(recon[i][0], cmap='gray')
        plt.axis('off')
    # Display the figure
    plt.show()
