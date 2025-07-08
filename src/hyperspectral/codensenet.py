from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class route_func3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16, activation='sigmoid'):
        super().__init__()

        self.activation = activation
        self.num_experts = num_experts
        self.out_channels = out_channels

        # Global Average Pool for 3D
        self.gap1 = nn.AdaptiveAvgPool3d(1)
        self.gap3 = nn.AdaptiveAvgPool3d(3)

        squeeze_channels = max(in_channels // reduction, reduction)

        self.dwise_separable = nn.Sequential(
            nn.Conv3d(2 * in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze_channels, num_experts * out_channels, kernel_size=1, stride=1, groups=1, bias=False)
        )

        if self.activation == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            self.temperature = 16
            self.activation_func = nn.Softmax(dim=2)

    def forward(self, x):
        b, _, _, _, _ = x.size()  # 3D input: [batch, channels, depth, height, width]
        a1 = self.gap1(x)
        a3 = self.gap3(x)
        a1 = a1.expand_as(a3)
        attention = torch.cat([a1, a3], dim=1)
        if self.activation == 'sigmoid':
            attention = self.activation_func(self.dwise_separable(attention))
        else:
            attention = self.dwise_separable(attention).view(b, self.num_experts, self.out_channels)
            attention = self.activation_func(attention * self.temperature).view(b, -1, 1, 1, 1)
        return attention


class route_func_single_scale3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16):
        super().__init__()
        # Global Average Pool for 3D
        self.gap1 = nn.AdaptiveAvgPool3d(1)

        squeeze_channels = max(in_channels // reduction, reduction)

        self.dwise_separable = nn.Sequential(
            nn.Conv3d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze_channels, squeeze_channels, kernel_size=1, stride=1, groups=squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze_channels, num_experts * out_channels, kernel_size=1, stride=1, groups=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _, _ = x.size()  # 3D input
        a1 = self.gap1(x)
        attention = self.sigmoid(self.dwise_separable(a1))
        return attention


class CoConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1,
                 reduction=16, bias=False, fuse_conv=False, activation='sigmoid'):
        super().__init__()
        self.fuse_conv = fuse_conv
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Routing function updated to 3D
        self.routing_func = route_func3d(in_channels, out_channels, num_experts, reduction, activation)

        if fuse_conv:
            self.convs = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size,
                                                   kernel_size, kernel_size))  # 3D kernel
            nn.init.kaiming_uniform_(self.convs, a=math.sqrt(5))

            if bias:
                self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
            else:
                self.register_parameter('bias', None)
            self.bns = nn.BatchNorm3d(out_channels)  # 3D batch norm
        else:
            self.convs = nn.ModuleList([nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                                  padding=padding, groups=groups, bias=bias) for _ in
                                        range(num_experts)])
            self.bns = nn.ModuleList([nn.BatchNorm3d(out_channels) for _ in range(num_experts)])

    def forward(self, x):
        routing_weight = self.routing_func(x)  # N x k*C
        if self.fuse_conv:
            routing_weight = routing_weight.view(-1, self.num_experts, self.out_channels).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1)  # Add extra dimension for depth
            b, c_in, d, h, w = x.size()  # 3D input
            x = x.view(1, -1, d, h, w)
            weight = self.convs.unsqueeze(0)

            combined_weight = (weight * routing_weight).view(self.num_experts, b * self.out_channels,
                                                             c_in // self.groups, self.kernel_size, self.kernel_size,
                                                             self.kernel_size)
            combined_weight = torch.sum(combined_weight, dim=0)
            if self.bias is not None:
                combined_bias = routing_weight.squeeze(-1).squeeze(-1).squeeze(-1).view(-1,
                                                                                        self.num_experts * self.out_channels) * self.bias.view(
                    -1).unsqueeze(0)
                combined_bias = combined_bias.sum(1)
                output = F.conv3d(x, weight=combined_weight, bias=combined_bias,
                                  stride=self.stride, padding=self.padding, groups=self.groups * b)
            else:
                output = F.conv3d(x, weight=combined_weight,
                                  stride=self.stride, padding=self.padding, groups=self.groups * b)
            output = self.bns(output.view(b, self.out_channels, output.size(-3), output.size(-2), output.size(-1)))
        else:
            outputs = []
            for i in range(self.num_experts):
                route = routing_weight[:, i * self.out_channels:(i + 1) * self.out_channels]
                out = self.convs[i](x)
                out = self.bns[i](out)
                out = out * route.expand_as(out)
                outputs.append(out)
            output = sum(outputs)
        return output


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=False, groups=groups))


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseLayer, self).__init__()
        # 1x1 conv: i --> bottleneck * k
        # self.conv_1 = DynamicMultiHeadConv(in_channels, bottleneck * growth_rate, kernel_size=1, heads=heads,
        #                                    squeeze_rate=squeeze_rate, gate_factor=gate_factor)
        # self.conv_1 = Dynamic_conv3d(in_channels, bottleneck * growth_rate, kernel_size=1, ratio=0.25, stride=1,
        #                              padding=0, dilation=1, groups=1,
        #                              bias=True, K=4, temperature=34)
        self.conv_1 = CoConv3d(in_channels, bottleneck * growth_rate, kernel_size=1, num_experts=heads, groups=1,
                             reduction=16, activation='sigmoid')

        # 3x3 conv: bottleneck * k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)
        # self.conv_2 = Dynamic_conv3d(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x_)
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)
        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, gate_factor, squeeze_rate,
                                group_3x3, heads)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CodenseNet(nn.Module):
    def __init__(self, band, num_classes):
        super(CodenseNet, self).__init__()
        self.name = 'codensenet'
        self.stages = [14, 14, 14]
        self.growth = [8, 16, 32]
        # self.stages = [14, 14, 14]
        # self.growth = [8, 16, 32]
        # self.stages = [4, 6, 8]
        # self.growth = [8, 16, 32]
        # self.stages = [10, 10, 10]
        # self.growth = [8, 16, 32]
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 7
        self.bottleneck = 4
        self.gate_factor = 0.25
        self.squeeze_rate = 16
        self.group_3x3 = 4
        self.heads = 4

        self.features = nn.Sequential()
        # Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        # Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv3d(1, self.num_features, kernel_size=3, stride=self.init_stride,
                                                        padding=1, bias=False))
        for i in range(len(self.stages)):
            # Dense-block i
            self.add_block(i)

        # Linear layer
        self.bn_last = nn.BatchNorm3d(self.num_features)
        self.relu_last = nn.ReLU(inplace=True)
        # self.pool_last = nn.AvgPool3d(self.pool_size)
        self.pool_last = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.classifier.bias.data.zero_()

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def add_block(self, i):
        # Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            bottleneck=self.bottleneck,
            gate_factor=self.gate_factor,
            squeeze_rate=self.squeeze_rate,
            group_3x3=self.group_3x3,
            heads=self.heads
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)

    def forward(self, x, progress=None, threshold=None):
        # if progress:
        #     DynamicMultiHeadConv.global_progress = progress
        features = self.features(x)
        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = CodenseNet(200, 12)

    from torchsummary import summary

    summary(net, input_size=[(1, 200, 7, 7)], batch_size=1)

    from thop import profile

    input = torch.randn(1, 1, 200, 7, 7)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
