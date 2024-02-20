import torch.nn as nn
import torch
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        
    def forward(self, img):
        return self.feature_extractor(img)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8)
        )
    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorSlowNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorSlowNet, self).__init__()
        
        #input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU()
        )
        
        #residual blocks "B"
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        #Post Res Blocks conv Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64,0.8)
        )
        
        #Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
        #Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), 
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out  = self.res_blocks(out1)
        out2 = self.conv2(out)
        out  = torch.add(out1, out2)
        out  = self.upsampling(out)
        out  = self.conv3(out)
        return out
    
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, vector_size=32):
        super(GeneratorResNet, self).__init__()
        
        #input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU()
        )
        
        #residual blocks "B"
        pre_res_blocks = []
        post_res_blocks = []
        
        pre_vector_count  = 8
        post_vector_count = 8
        
        meta_size = vector_size + 64
        
        for _ in range(pre_vector_count):
            pre_res_blocks.append(ResidualBlock(64))
        for _ in range(post_vector_count):
            post_res_blocks.append(ResidualBlock(meta_size))

        self.pre_res_blocks  = nn.Sequential(*pre_res_blocks)
        self.post_res_blocks = nn.Sequential(*post_res_blocks)
        
        #Post Res Blocks conv Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(meta_size, meta_size, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(meta_size, 0.8)
        )
        #Upsampling layers
        upsample_size = meta_size * 4
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(meta_size, upsample_size, 3, 1, 1),
                nn.BatchNorm2d(upsample_size),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
        #Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(meta_size, out_channels, kernel_size=9, stride=1, padding=4), 
            nn.Sigmoid()
        )
        
    def forward(self, img, meta):
        #Shape of nx64x64x64
        pre_skip = self.conv1(img)
        pre_vec  = self.pre_res_blocks(pre_skip)
        #Shape of nx96x64x64
        with_vec = torch.cat((pre_vec, meta), 1)
        post_vec = self.post_res_blocks(with_vec)
        to_skip  = self.conv2(post_vec)
        #Reshape from nx64x64x64 to nx96x64x64 by appending metadata
        pre_skip = torch.cat((pre_skip, meta), 1)
        out      = torch.add(pre_skip, to_skip)
        out      = self.upsampling(out)
        out      = self.conv3(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64,128,256,512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)
    
class MetaAutoencoderTail(nn.Module):
    def __init__(self):
        #N,5,512,512
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1), #Nx5x256x256 -> Nx16x256x256
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Nx16x256x256 -> Nx32x128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), #Nx32x128x128 -> Nx64x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #include the decoder so all state dict key match
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), #Nx64x128x128 -> Nx32x256x256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), #Nx32x256x256 -> Nx16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), #Nx32x256x256 -> Nx16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 5, kernel_size=3, stride=1, padding=1, bias=False), #Nx16x512x512 -> Nx5x512x512
            nn.Sigmoid(),
        )
    def forward(self, x):
        vec = self.encoder(x)
        return vec

class MetaAutoencoder(nn.Module):
    def __init__(self):
        #N,5,512,512
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1), #Nx5x256x256 -> Nx16x256x256
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Nx16x256x256 -> Nx32x128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), #Nx32x128x128 -> Nx64x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), #Nx64x128x128 -> Nx32x256x256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), #Nx32x256x256 -> Nx16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), #Nx32x256x256 -> Nx16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 5, kernel_size=3, stride=1, padding=1, bias=False), #Nx16x512x512 -> Nx5x512x512
            nn.Sigmoid(),
        )
    def forward(self, x):
        vec = self.encoder(x)
        out = self.decoder(vec)
        return out
