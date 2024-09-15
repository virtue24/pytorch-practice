import torch
import cv2
import numpy as np

class KernelFilter(torch.nn.Module):
    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        self.kernel = torch.nn.Parameter(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, self.kernel, padding=1)  # padding=1 to keep the output size same as input

# Read and convert the image to RGB


epochs = 1000
lowest_mse = float('inf')
best_kernel = None

for epoch in range(epochs):
    image = cv2.imread("toothbrush/train/good/000.png")
    # cv2.imshow("Image", image)
    # cv2.waitKey(250)

    # Convert the image to a tensor and add batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Create a kernel for RGB channels (3 output channels, 3 input channels, n x n size)
    # This example uses a 3x3 kernel for each channel, but you can modify it for any nxn kernel
    kernel = torch.randn(3, 3, 3, 3, dtype=torch.float32)

    # Normalize each 3x3 filter in the kernel so that the sum of values at each (output_channel, input_channel) slice sums to 1
    # Use the sum of each 3x3 filter for normalization
    for i in range(kernel.size(0)):  # iterate over output channels
        for j in range(kernel.size(1)):  # iterate over input channels
            kernel[i, j] /= kernel[i, j].sum()

    # Now each 3x3 filter in the kernel will sum to 1

    # Kernel shape: (output_channels, input_channels, kernel_height, kernel_width)
    # In this case: (3, 3, 3, 3)
    model = KernelFilter(kernel)

    # Forward pass
    output = model(image_tensor)

    # Compute the mean of the output
    output_mean = torch.mean(output)

    # Compute the MSE: (output - mean)^2
    mse = torch.mean((output - output_mean) ** 2).item()

     # Update the best kernel based on the lowest MSE
    if mse < lowest_mse:
        print(f"Epoch: {epoch}, MSE: {mse}")

        lowest_mse = mse
        best_kernel = kernel

        output = output.squeeze().permute(1, 2, 0).detach().numpy()

        # Normalize the output image so that pixel values range from 0 to 255
        min_val = np.min(output)
        max_val = np.max(output)
        output = (output - min_val) / (max_val - min_val) * 255
        output = np.clip(output, 0, 255).astype(np.uint8)

        # Display the normalized output image
        cv2.imshow("Normalized Output", output)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
