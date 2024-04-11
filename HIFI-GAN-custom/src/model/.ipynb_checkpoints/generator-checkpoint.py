from torch import nn
from src.utils import init_weights


class ResBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_size,
                 dilations):
        super().__init__()

        layers = []
        for i in range(len(dilations)):
            block = []
            for j in range(len(dilations[0])):
                block.append(nn.LeakyReLU(0.1))
                conv_layer = nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    padding='same',
                    dilation=dilations[i][j]
                )
                conv_layer.apply(init_weights)
                block.append(nn.utils.weight_norm(
                    conv_layer
                ))
            layers.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class MRF(nn.Module):
    def __init__(self,
                 num_channels,
                 mrf_kernel_sizes,
                 mrf_dilations):
        super().__init__()

        blocks = []
        for i in range(len(mrf_dilations)):
            blocks.append(
                ResBlock(
                    num_channels=num_channels,
                    kernel_size=mrf_kernel_sizes[i],
                    dilations=mrf_dilations[i]
                )
            )
        self.residual_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        sum_outs = self.residual_blocks[0](x)
        for res_block in self.residual_blocks[1:]:
            sum_outs += res_block(x)
        return sum_outs


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_transpose_kernel_size,
                 mrf_kernel_sizes,
                 mrf_dilations):
        super().__init__()
        conv_transpose = nn.utils.weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=conv_transpose_kernel_size,
                stride=conv_transpose_kernel_size // 2,
                padding=(conv_transpose_kernel_size - conv_transpose_kernel_size // 2) // 2
            )
        )
        conv_transpose.apply(init_weights)
        self.block = nn.Sequential(
            nn.LeakyReLU(0.1),
            conv_transpose,
            MRF(
                num_channels=in_channels // 2,
                mrf_kernel_sizes=mrf_kernel_sizes,
                mrf_dilations=mrf_dilations
            )
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 conv_transpose_kernel_sizes,
                 mrf_kernel_sizes,
                 mrf_dilations):
        super().__init__()

        self.to_hidden = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=7,
                dilation=1,
                padding='same'
            )
        )

        self.gen_blocks = nn.ModuleList([
            GeneratorBlock(
                in_channels=hidden_dim // 2 ** i,
                conv_transpose_kernel_size=conv_transpose_kernel_sizes[i],
                mrf_kernel_sizes=mrf_kernel_sizes,
                mrf_dilations=mrf_dilations
            )
            for i in range(len(conv_transpose_kernel_sizes))
        ])
        conv_from_hidden = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=hidden_dim // (2 ** (len(conv_transpose_kernel_sizes))),
                out_channels=1,
                kernel_size=7,
                padding='same'
            )
        )
        conv_from_hidden.apply(init_weights)
        self.from_hidden = nn.Sequential(
            nn.LeakyReLU(0.1),
            conv_from_hidden,
            nn.Tanh()
        )

    def forward(self, x):
        x = self.to_hidden(x)
        for block in self.gen_blocks:
            x = block(x)
        x = self.from_hidden(x)
        return x
    
    def remove_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose1d):
                nn.utils.remove_weight_norm(module)
