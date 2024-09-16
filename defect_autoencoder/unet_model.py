import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.contract_block(in_channels, 64)
        self.enc2 = self.contract_block(64, 128)
        self.enc3 = self.contract_block(128, 256)
        self.enc4 = self.contract_block(256, 512)
        self.enc5 = self.contract_block(512, 1024)

        # Expanding path (Decoder)
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.contract_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.contract_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.contract_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.contract_block(128, 64)

        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop_tensor(self, enc_feature, target_size):
        """
        Crops the enc_feature tensor to match the size of the target_size tensor.
        This is necessary because during the upsampling and downsampling,
        the sizes might not match perfectly, so we need to crop.
        """
        _, _, h, w = enc_feature.size()
        target_h, target_w = target_size[2], target_size[3]

        delta_h = h - target_h
        delta_w = w - target_w

        # Crop along height and width
        enc_feature = enc_feature[:, :, delta_h // 2 : h - delta_h // 2, delta_w // 2 : w - delta_w // 2]
        return enc_feature

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(enc5)
        enc4 = self.crop_tensor(enc4, dec4.size())  # Crop the encoder feature map to match the decoder
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.crop_tensor(enc3, dec3.size())
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.crop_tensor(enc2, dec2.size())
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.crop_tensor(enc1, dec1.size())
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)

if __name__ == "__main__":
    # Example of using the UNet model
    model = UNet(in_channels=3, out_channels=1)
    output = model(torch.randn(1, 3, 572, 572))  # Example input tensor (batch_size, channels, height, width)
    print(output.shape)  # Output shape: torch.Size([1, 1, 388, 388])
