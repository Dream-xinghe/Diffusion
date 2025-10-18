import torch.nn as nn
from timm.models.layers import Mlp


#实现MLP-Mixer模块,用纯 MLP 交替在“token 维度”和“通道维度”上做混合，替代自注意力完成信息交互。
class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, channels)
        # 分别对token和channel维度进行归一化和MLP操作
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        # Token-mixing MLP
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        # return的size是 (batch_size, num_tokens, channels)
        return x + self.channels_mlp(y)