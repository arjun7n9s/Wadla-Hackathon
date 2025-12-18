
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, 3, 256, 256)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> (32, 128, 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (256, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: (B, 256, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (128, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (32, 128, 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (3, 256, 256)
            nn.Sigmoid() # Pixel values 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
