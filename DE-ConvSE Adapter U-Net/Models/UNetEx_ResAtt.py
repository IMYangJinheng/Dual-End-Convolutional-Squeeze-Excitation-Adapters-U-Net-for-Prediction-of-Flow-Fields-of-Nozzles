import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as WN


# ========== 小工具函数：创建卷积层 ==========
def create_layer(in_channels, out_channels, kernel_size=3,
                 wn=True, bn=True, activation=nn.ReLU, convolution=nn.Conv2d):
    assert kernel_size % 2 == 1, "kernel_size 必须是奇数以保持尺寸"
    layers = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = WN(conv)
    layers.append(conv)
    if activation is not None:
        layers.append(activation())
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


# ========== SE 注意力块 ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


# ========== 残差块 ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 wn=True, bn=True, activation=nn.ReLU):
        super().__init__()
        self.conv1 = create_layer(in_channels, out_channels, kernel_size, wn=wn, bn=bn, activation=activation)
        self.conv2 = create_layer(out_channels, out_channels, kernel_size, wn=wn, bn=bn, activation=None)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(x)
        return self.act(out)


# ========== 编码器块 ==========
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 wn=True, bn=True, activation=nn.ReLU, layers=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels if i == 0 else out_channels,
                            out_channels, kernel_size, wn, bn, activation)
              for i in range(layers)]
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.blocks(x)
        return self.se(x)


# ========== 解码器块 ==========
class DecoderBlock(nn.Module):
    def __init__(self, skip_channels, out_channels, kernel_size=3,
                 wn=True, bn=True, activation=nn.ReLU, layers=2):
        super().__init__()
        layers_list = []

        # 关键修复：明确参数含义
        # skip_channels: skip connection 的通道数
        # out_channels: 解码器输出通道数
        # 拼接后输入通道数 = skip_channels * 2
        first_layer_in_channels = skip_channels * 2

        layers_list.append(create_layer(first_layer_in_channels, out_channels, kernel_size,
                                        wn=wn, bn=bn, activation=activation))

        # 后续层
        for i in range(1, layers):
            layers_list.append(create_layer(out_channels, out_channels, kernel_size,
                                            wn=wn, bn=bn, activation=activation))

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)


# ========== UNetEx_ResAtt 主体 ==========
class UNetEx_ResAtt(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, filters=[16, 32, 64],
                 layers=2, batch_norm=True, weight_norm=True,
                 activation=nn.ReLU, final_activation=None):
        super().__init__()
        self.final_activation = final_activation
        self.filters = filters

        # 编码器
        self.encoders = nn.ModuleList()
        for i, f in enumerate(filters):
            self.encoders.append(
                EncoderBlock(in_channels if i == 0 else filters[i - 1],
                             f, kernel_size, weight_norm, batch_norm, activation, layers)
            )

        # 解码器（逆序）- 关键修复：使用正确的参数
        # 解码器（逆序）
        self.decoders = nn.ModuleList()
        for i in range(len(filters) - 1):
            skip_channels = filters[-(i + 2)]
            dec_out_channels = filters[-(i + 2)]  # ✅ 改名，避免覆盖真正的 out_channels

            self.decoders.append(
                DecoderBlock(skip_channels, dec_out_channels, kernel_size,
                             weight_norm, batch_norm, activation, layers)
            )

        # 最后输出层
        self.final_conv = create_layer(filters[0], out_channels, kernel_size,
                                       wn=weight_norm, bn=False, activation=final_activation)

        # 预定义通道调整层
        self.adjust_layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.adjust_layers.append(
                nn.Conv2d(filters[-(i + 1)], filters[-(i + 2)], kernel_size=1)
            )

    def encode(self, x):
        tensors, indices, sizes = [], [], []
        for encoder in self.encoders:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            if encoder is not self.encoders[-1]:
                x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
                indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, x, tensors, indices, sizes):
        tensors = list(reversed(tensors))
        indices = list(reversed(indices))
        sizes = list(reversed(sizes))

        for i, decoder in enumerate(self.decoders):
            ind = indices[i]
            out_size = sizes[i + 1] if i + 1 < len(sizes) else sizes[-1]
            skip = tensors[i + 1]

            # 调整通道数以匹配索引
            if x.shape[1] != ind.shape[1]:
                x = self.adjust_layers[i](x)

            # 反池化
            x = F.max_unpool2d(x, ind, kernel_size=2, stride=2, output_size=out_size)

            # 拼接
            x = torch.cat([x, skip], dim=1)

            # 解码
            x = decoder(x)

        # 最终输出
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x_enc, tensors, indices, sizes = self.encode(x)
        x = self.decode(x_enc, tensors, indices, sizes)
        return x

# model = UNetEx_ResAtt(in_channels=2, out_channels=3, filters=[8, 16, 32, 64, 128])
# x = torch.randn(1, 2, 111, 381)
# out = model(x)
# print(out.shape)

