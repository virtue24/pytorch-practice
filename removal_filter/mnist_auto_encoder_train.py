import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print('Image tensor min and max values:', torch.min(images), torch.max(images))

# Define the Autoencoder model
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

# **Add device code check**
# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize the model and move it to the device
model = Autoencoder().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3,
                             weight_decay=1e-5)

# Training loop
num_epochs = 75
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
torch.save(model.state_dict(), 'autoencoder_mnist.pth')
print('Model saved as autoencoder_mnist.pth')

# Plotting original and reconstructed images
for k in range(0, num_epochs, 1):
    plt.figure(figsize=(9, 2))
    plt.gray()
    # Move tensors to CPU before converting to NumPy
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    # Plot original images
    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i + 1)
        plt.imshow(item[0], cmap='gray')
        plt.axis('off')
    # Plot reconstructed images
    for i, item in enumerate(recon):
        if i >= 9:
            break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0], cmap='gray')
        plt.axis('off')
    # Display the figure
    plt.show()
    