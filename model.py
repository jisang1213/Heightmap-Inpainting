class DoublePConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoublePConv, self).__init__()

        self.pconv1 = PConv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pconv2 = PConv2d(out_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        x, mask = self.pconv1(x, mask)
        x = self.relu(self.batchnorm(x))
        x, mask = self.pconv2(x, mask)
        x = self.relu(self.batchnorm(x))
        return x, mask

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], #height and width of layers. Change in channels to 1 for greyscale
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoublePConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,        #Change to nearest neighbout upsampling
                )
            )
            self.ups.append(DoublePConv(feature*2, feature))

        self.bottleneck = DoublePConv(features[-1], features[-1]*2)
        self.final_conv = PConv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, mask):
        skip_connections = []
        masks = []

        nextmask = mask

        for down in self.downs:
            x, nextmask = down(x, nextmask)
            skip_connections.append(x)
            masks.append(nextmask)
            x = self.pool(x)
            nextmask = self.pool(nextmask)

        x, nextmask = self.bottleneck(x, nextmask)
        skip_connections = skip_connections[::-1]
        masks = masks[::-1]

        for idx in range(0, len(self.ups), 2):
            print(type(x))
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            nextmask = masks[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                nextmask = TF.resize(mask, size=mask.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x, nextmask = self.ups[idx+1](concat_skip, nextmask)

        output, mask_placeholder = self.final_conv(x, mask)   #convolve with initial mask

        return output

def test():
    x = torch.randn((3, 1, 161, 161))
    mask = (torch.rand(3, 161, 161) > 0.5).to(torch.float32)
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x, mask)
    assert preds.shape == x.shape
    print(x.shape)
    print(preds.shape)

if __name__ == "__main__":
    test()