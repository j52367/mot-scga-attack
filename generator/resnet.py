import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
# Weight Initialization
###########################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

###########################
# Residual Block
###########################

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        return x + self.block(x)

###########################
# Generator Architecture
###########################

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False):
        super().__init__()
        self.ngf = 64
        self.inception = inception

        # Downsampling
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, self.ngf, 7, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(self.ngf*2, self.ngf*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.ngf*4) for _ in range(6)]
        )

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 3, stride=2, 
                              padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf*2, self.ngf, 3, stride=2,
                              padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf, 3, 7, padding=0),
            nn.Tanh()
        )

        # Initialization
        self.apply(weights_init_normal)

    def forward(self, x, feat=False):
        # Encoder
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Residual features
        features = self.res_blocks(x)
        
        # Decoder
        x = self.upsample1(features)
        x = self.upsample2(x)
        output = self.output(x)
        
        # Inception compatibility
        if self.inception:
            output = output[:, :, :-1, :-1]  # Crop to 299x299
        
        return (output, features) if feat else output

###########################
# Testing Module
###########################

if __name__ == '__main__':
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    img_size = 256
    
    # Initialize components
    netG = GeneratorResnet().to(device)
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        output, features = netG(dummy_input, return_features=True)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
