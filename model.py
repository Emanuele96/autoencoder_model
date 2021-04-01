import torch 
import torch.nn as nn
import math

def calculate_conv2d_output(input_size, stride, padding, dilatation, kernel_size):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    #format (height, width)
    height_output = math.floor((input_size[0] + 2 * padding[0] - dilatation[0] * (kernel_size[0] - 1)/ stride[0]) + 1) 
    width_output = math.floor((input_size[1] + 2 * padding[1] - dilatation[1] * (kernel_size[1] - 1)/ stride[1]) + 1) 
    return height_output, width_output

class Autoencoder(nn.Module):

    def __init__(self, input_shape):
        super(Net, self ).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=1),
            nn.ReLU,
            nn.MaxPool2d(kernel_size= 2, stride=2)
        )

        self.latent_vector = nn.Sequential(
            nn.Linear(in_features=10*2*2, out_features = 10)
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels= 1, out_channels = 4, kernel_size = 5, stride= 1, padding= 1),
            nn.ReLU,
            nn.ConvTranspose2d(in_channels= 4, out_channels = 1, kernel_size = 5, stride= 1, padding= 1),
            nn.ReLU()
        )


    def forward(self, x):
        #forward through the encoder first
        x = self.encoder(x)
        #reshape and pass through the latent vector
        x = x.view(x.size(0), -1)
        x = self.latent_vector(x)
        #reshape and pass throught the decoder
        x = x.view(1, 5, 5)
        x = self.decoder(x)
        return x
