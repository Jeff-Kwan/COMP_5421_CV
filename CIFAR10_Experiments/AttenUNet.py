import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels, classes):
        super(ResBlock, self).__init__()
        self.num_blocks = 4
        self.channels = channels
        self.blocks = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, 1, 1),
            nn.GroupNorm(1, channels*2),
            nn.SiLU(),
            nn.Conv2d(channels*2, channels, 3, 1, 1, bias=False))
        for _ in range(self.num_blocks)])
        self.norm = nn.ModuleList([nn.GroupNorm(1, channels, affine=False) for _ in range(self.num_blocks)])
        self.c_embed = nn.Embedding(classes, 2*channels)

    def forward(self, x, c):
        c_mu, c_logvar = self.c_embed(c).view(-1, self.channels*2, 1, 1).chunk(2, dim=1)
        c_std = torch.exp(0.5 * c_logvar)
        x = c_std * x + c_mu
        for i in range(self.num_blocks):
            x = self.norm[i](x + self.blocks[i](x))
        return x

class AttenUNet(nn.Module):
    def __init__(self, in_c, out_c, layers=3, channels=24):
        super(AttenUNet, self).__init__()
        self.layers = layers - 1    # Original resolution is 1 layer
        self.channels = channels
        self.classes = 10

        self.in_conv = nn.Conv2d(in_c, self.channels, 3, 1, 1, bias=False)
        self.in_norm = nn.GroupNorm(1, self.channels, affine=False)
        self.t_embed = nn.Linear(2, self.channels)

        # Encoder 
        self.encoder_blocks = nn.ModuleList([ResBlock(self.channels, self.classes) for _ in range(self.layers)])
        self.downs = nn.ModuleList([nn.Conv2d(self.channels, self.channels, 2, 2, 0, bias=None) 
                                    for _ in range(self.layers)])

        # Bottleneck
        self.attention = nn.ModuleList([nn.MultiheadAttention(self.channels, 2, bias=False) for _ in range(2)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(self.channels, self.channels*4),
            nn.SiLU(),
            nn.Linear(self.channels*4, self.channels, bias=None)) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.channels, elementwise_affine=False, bias=False) for _ in range(2*2)])
        self.c_embed = nn.Embedding(self.classes, 2*channels)
        

        # Decoder
        self.ups = nn.ModuleList([nn.ConvTranspose2d(self.channels, self.channels, 2, 2, 0, bias=False)
                        for _ in range(self.layers)])
        self.merges = nn.ModuleList([nn.Conv2d(self.channels*2, self.channels, 1, 1, 0, bias=False)
                        for _ in range(self.layers)])
        self.decoder_blocks = nn.ModuleList([ResBlock(self.channels, self.classes) for _ in range(self.layers)])

        self.out_norm = nn.GroupNorm(1, self.channels, affine=False)
        self.out = nn.Conv2d(self.channels, out_c, kernel_size=1, stride=1, padding=0)
        
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
        B, C, H, W = x.size()
        c_mu, c_logvar = self.c_embed(c).unsqueeze(1).chunk(2, dim=-1)
        c_std = torch.exp(0.5 * c_logvar)
        x = x.view(B, C, H*W).transpose(1,2).contiguous()
        x = c_std * x + c_mu
        for i in range(2):
            x = self.norms[2*i](x + self.attention[i](x, x, x)[0])
            x = self.norms[2*i+1](x + self.ffn[i](x))
        x = x.transpose(1,2).view(B, C, H, W)

        # Decoder
        for i in range(self.layers):
            x = self.ups[i](x)
            x = self.merges[i](torch.cat([x, skips[-i-1]], dim=1))
            x = self.decoder_blocks[i](x, c)

        x = self.out_norm(x)
        x = self.out(x)
        return x