import torch
from torch import nn

class MHA2D(nn.Module):
    def __init__(self, channels, heads=4, bias=False):
        super(MHA2D, self).__init__()
        self.layernorm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True, bias=bias)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).transpose(1, 2).contiguous()
        x = self.layernorm(x)
        x = self.mha(x, x, x, need_weights=False)[0]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, heads, classes):
        super(ResBlock, self).__init__()
        self.num_blocks = 2
        self.channels = channels
        self.mha = nn.ModuleList([
            MHA2D(channels, heads)
            for _ in range(self.num_blocks)])
        self.convblocks = nn.ModuleList([
        nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.Conv2d(channels, channels*2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels*2, channels, 3, 1, 1))
        for _ in range(self.num_blocks)])
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.c_embed = nn.Embedding(classes, 2*channels)

    def forward(self, x, c=None):
        if c is not None:
            c_mu, c_logvar = self.c_embed(c).view(-1, self.channels*2, 1, 1).chunk(2, dim=1)
            c_std = torch.exp(0.5 * c_logvar)
            x = c_std * x + c_mu
        for i in range(self.num_blocks):
            x = x + self.mha[i](x)
            x = x + self.convblocks[i](x)
        x = self.norm(x)
        return x

class AttenUNet(nn.Module):
    def __init__(self, layers=3, channels=24, heads=4):
        super(AttenUNet, self).__init__()
        self.layers = layers - 1    # Original resolution is 1 layer
        self.channels = channels
        self.classes = 10

        self.in_conv = nn.Conv2d(1, self.channels, 3, 1, 1, bias=False)
        self.in_norm = nn.GroupNorm(1, self.channels, affine=False)
        self.t_embed = nn.Linear(2, self.channels)

        # Encoder 
        self.encoder_blocks = nn.ModuleList([ResBlock(self.channels * (2 ** i), heads*2**(i//2), self.classes) for i in range(self.layers)])
        self.downs = nn.ModuleList([nn.Conv2d(self.channels * (2 ** i), self.channels * (2 ** (i + 1)), 2, 2, 0, bias=None) 
                                    for i in range(self.layers)])

        # Bottleneck
        self.bottleneck = ResBlock(self.channels * (2 ** self.layers), heads*2**(self.layers//2), self.classes)
        
        # Decoder
        self.ups = nn.ModuleList([nn.ConvTranspose2d(self.channels * (2 ** (i + 1)), self.channels * (2 ** i), 2, 2, 0, bias=False)
                        for i in reversed(range(self.layers))])
        self.merges = nn.ModuleList([nn.Conv2d(self.channels * (2 ** (i + 1)), self.channels * (2 ** i), 1, 1, 0, bias=False)
                        for i in reversed(range(self.layers))])
        self.decoder_blocks = nn.ModuleList([ResBlock(self.channels * (2 ** i), heads*2**(i//2), self.classes) for i in reversed(range(self.layers))])

        self.out_norm = nn.GroupNorm(1, self.channels, affine=False)
        self.out = nn.Conv2d(self.channels, 1, 1, 1, 0)
        
    def forward(self, x, t, c):
        x = self.in_conv(x)
        x = x + self.t_embed(torch.stack([t, 1-t], dim=-1)).view(-1, self.channels, 1, 1)
        x = self.in_norm(x)

        # Encoder
        skips = []
        for i in range(self.layers):
            x = self.encoder_blocks[i](x, c)
            skips.append(x)
            x = self.downs[i](x)

        # Bottleneck
        x = self.bottleneck(x, c)

        # Decoder
        for i in range(self.layers):
            x = self.ups[i](x)
            x = self.merges[i](torch.cat([x, skips[-i-1]], dim=1))
            x = self.decoder_blocks[i](x, c)

        x = self.out_norm(x)
        x = self.out(x)
        return x
    

if __name__ == "__main__":
    # Example usage
    model = AttenUNet(layers=3, channels=24)
    x = torch.randn(8, 3, 32, 32)  # Batch of 8 images
    t = torch.tensor([0.5] * 8)  # Time step
    c = torch.randint(0, 10, (8,))  # Random class labels
    output = model(x, t, c)
    print(output.shape)  # Should be (8, 3, 32, 32)