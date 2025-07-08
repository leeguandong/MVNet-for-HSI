from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# 移除mamba_ssm依赖，用纯PyTorch实现Mamba核心机制
class SimpleMambaLayer(nn.Module):
    """简化版Mamba层，去除mamba_ssm依赖"""

    def __init__(self, d_model, d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 线性投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # 1D卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 状态空间参数
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        x: (B, L, D)
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # 转换为1D卷积格式
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]  # 移除padding
        x = rearrange(x, "b d l -> b l d")

        # 激活函数
        x = F.silu(x)
        z = F.silu(z)

        # 简化的状态空间计算
        # 这里用简化版本替代复杂的selective scan
        ssm_out = self._simple_ssm(x)

        # 门控机制
        out = ssm_out * z

        # 输出投影
        out = self.out_proj(out)
        return out

    def _simple_ssm(self, x):
        """简化的状态空间模型计算"""
        batch_size, seq_len, d_inner = x.shape

        # 使用可学习的投影而不是复杂的selective scan
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            # 简化的状态更新
            u = x[:, t, :]  # (B, d_inner)
            h = h + torch.einsum('bd,ds->bds', u, self.B)  # 状态更新
            y = torch.einsum('bds,ds->bd', h, self.C) + u * self.D  # 输出
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class HSI_MambaVisionMixer(nn.Module):
    """适用于高光谱影像的MambaVision混合器"""

    def __init__(self, d_model, d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.mamba_layer = SimpleMambaLayer(d_model, d_state, d_conv, expand)

    def forward(self, hidden_states):
        return self.mamba_layer(hidden_states)


class HSI_Attention(nn.Module):
    """适用于高光谱的注意力机制"""

    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HSI_MLP(nn.Module):
    """MLP层"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HSI_MambaVisionBlock(nn.Module):
    """高光谱MambaVision块"""

    def __init__(self, dim, num_heads=4, counter=0, transformer_blocks=[],
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # 根据counter决定使用Mamba还是Attention
        if counter in transformer_blocks:
            self.mixer = HSI_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.mixer = HSI_MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = HSI_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HSI_Conv3dBlock(nn.Module):
    """3D卷积块，用于处理高光谱数据的空间-光谱特征"""

    def __init__(self, dim, kernel_size=3, drop_path=0.):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)
        self.norm1 = nn.BatchNorm3d(dim)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)
        self.norm2 = nn.BatchNorm3d(dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = identity + self.drop_path(x)
        return x


def window_partition_3d(x, window_size):
    """3D窗口分割"""
    B, C, D, H, W = x.shape
    x = x.view(B, C,
               D // window_size, window_size,
               H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size ** 3, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """3D窗口还原"""
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.reshape(B, D // window_size, H // window_size, W // window_size,
                        window_size, window_size, window_size, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, windows.shape[2], D, H, W)
    return x


class HSI_MambaVisionLayer(nn.Module):
    """高光谱MambaVision层"""

    def __init__(self, dim, depth, num_heads=4, window_size=7, conv=False,
                 downsample=True, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 transformer_blocks=[]):
        super().__init__()
        self.conv = conv
        self.window_size = window_size

        if conv:
            # 使用3D卷积块处理高光谱数据
            self.blocks = nn.ModuleList([
                HSI_Conv3dBlock(dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
                for i in range(depth)
            ])
            self.is_3d = True
        else:
            # 使用MambaVision块
            self.blocks = nn.ModuleList([
                HSI_MambaVisionBlock(
                    dim=dim, num_heads=num_heads, counter=i,
                    transformer_blocks=transformer_blocks, mlp_ratio=mlp_ratio,
                    qkv_bias=True, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
                for i in range(depth)
            ])
            self.is_3d = False

        # 下采样层
        if downsample:
            if conv:
                self.downsample = nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
            else:
                self.downsample = nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        if not self.conv:
            # 对于非卷积层，需要将3D转换为序列处理
            B, C, D, H, W = x.shape

            # 窗口分割
            if D % self.window_size != 0 or H % self.window_size != 0 or W % self.window_size != 0:
                pad_d = (self.window_size - D % self.window_size) % self.window_size
                pad_h = (self.window_size - H % self.window_size) % self.window_size
                pad_w = (self.window_size - W % self.window_size) % self.window_size
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                _, _, Dp, Hp, Wp = x.shape
            else:
                Dp, Hp, Wp = D, H, W
                pad_d = pad_h = pad_w = 0

            # 窗口分割
            x = window_partition_3d(x, self.window_size)  # (num_windows*B, window_size^3, C)

            # 通过MambaVision块
            for blk in self.blocks:
                x = blk(x)

            # 窗口还原
            x = window_reverse_3d(x, self.window_size, Dp, Hp, Wp)

            # 移除padding
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = x[:, :, :D, :H, :W].contiguous()
        else:
            # 3D卷积处理
            for blk in self.blocks:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class HSI_MambaVision(nn.Module):
    """高光谱MambaVision模型"""

    def __init__(self, num_bands=200, num_classes=16, spatial_size=7,
                 dim=80, in_dim=32, depths=[1, 3, 8, 4], window_size=[4, 4, 7, 7],
                 num_heads=[2, 4, 8, 16], mlp_ratio=4., drop_path_rate=0.2):
        super().__init__()

        # 输入嵌入层 - 适应高光谱数据
        self.patch_embed = nn.Sequential(
            nn.Conv3d(1, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_dim),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )

        # 计算drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建各层
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = HSI_MambaVisionLayer(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv,
                downsample=(i < len(depths) - 1),
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(
                    range(depths[i] // 2, depths[i]))
            )
            self.levels.append(level)

        # 分类头
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.norm = nn.BatchNorm3d(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x shape: (B, 1, bands, H, W)
        x = self.patch_embed(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 替换原有的DydenseNet
class MambaDenseNet(nn.Module):
    """高光谱MambaVision网络，用于替换DydenseNet"""

    def __init__(self, band=200, num_classes=16, spatial_size=7):
        super(MambaDenseNet, self).__init__()
        self.name = 'hsi_mambavision'

        # 使用最小的MambaVision-T配置
        self.model = HSI_MambaVision(
            num_bands=band,
            num_classes=num_classes,
            spatial_size=spatial_size,
            dim=80,  # MambaVision-T的基础维度
            in_dim=32,
            depths=[1, 3, 8, 4],  # MambaVision-T的深度配置
            window_size=[4, 4, 7, 7],  # 适应小尺寸高光谱patch
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.,
            drop_path_rate=0.2
        )

    def forward(self, x, progress=None, threshold=None):
        # x shape: (B, 1, bands, H, W)
        return self.model(x)


if __name__ == "__main__":
    # 测试模型
    net = HSI_MambaVisionNet(band=200, num_classes=16, spatial_size=7)

    # 创建测试输入
    input_tensor = torch.randn(1, 1, 200, 7, 7)
    output = net(input_tensor)
    print(f"输出形状: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f'参数量: {total_params / 1e6:.2f}M')

    # 计算FLOPs (需要安装thop)
    try:
        from thop import profile

        flops, params = profile(net, inputs=(input_tensor,))
        print(f'FLOPs: {flops / 1e9:.2f}GFLOPs')
    except ImportError:
        print("请安装thop库来计算FLOPs: pip install thop")