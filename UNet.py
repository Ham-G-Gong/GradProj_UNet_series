import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder = Block(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.decoder = Block(in_ch, out_ch)

    def forward(self, x, encoder_features):
        x = self.up(x)
        diffY = encoder_features.size()[2] - x.size()[2]
        diffX = encoder_features.size()[3] - x.size()[3]

        # Skip Connection!
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([encoder_features, x], dim=1)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = Block(3, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        self.encoder4 = Encoder(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)
        self.output = nn.Conv2d(64, 14, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        x = self.output(x)
        x = self.softmax(x)
        return x
