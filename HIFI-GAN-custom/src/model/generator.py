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


# EXPERIMENT (MOBILE-LIKE HIFI GENERATOR)
class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[2]), dilation=dilation[2])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            )
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class GeneratorMobileLike(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

