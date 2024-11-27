"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor = None
    ):
        return self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]


    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor = None,
    ):
        attn_output = self.attention(x=self.ln_1(x), attn_mask=attn_mask)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x



def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)




class Transformer_causal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.width = config.transformer.width
        self.num_layers = config.transformer.num_layers
        self.num_heads = config.transformer.num_heads
        self.num_layers_token = config.transformer.num_layers_token

        self.num_latent_tokens=256
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))


        # 初始化类别嵌入和位置嵌入
        scale = self.width ** -0.5  # 初始化缩放因子
        self.class_embedding = nn.Parameter(
            scale * torch.randn(1, self.width)  # 类别嵌入
        )
       

        self.ln_pre = nn.LayerNorm(self.width)  # 输入的归一化层
        self.ln_post = nn.LayerNorm(self.width)  # 输出的归一化层



        # 为每个 token 定义独立的 transformer 堆叠
        self.transformers_tokens = nn.ModuleList([
            nn.Sequential(*[
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
                for _ in range(self.num_layers_token)
            ])
            for _ in range(self.num_latent_tokens)
        ])

        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        latent_tokens=self.latent_tokens
        # [b,256(c),256(s)]
        x = pixel_values.reshape(batch_size, pixel_values.shape[1], -1)

        # [b,256(s),256(c)]
        x = x.permute(0, 2, 1) 

        # [b,257(s),256(c)]
        x = torch.cat([_expand_token(self.class_embedding, batch_size).to(x.dtype), x], dim=1)


        x_info = self.ln_pre(x)

  


        prev_tokens = []

        # 第三步：逐步生成 tokens
        for i in range(self.num_latent_tokens):
            # 获取当前的 latent token
            latent_token_i = _expand_token(latent_tokens[i], batch_size).to(x_info.dtype)  # 当前 token: [batch_size, 1, self.width]

            # 将当前 token 拼接到 x_info
            x = torch.cat([x_info, latent_token_i], dim=1)  # 拼接后: [batch_size, seq_length + 1, self.width]

            x = self.ln_pre(x)  # 归一化
            x = x.permute(1, 0, 2)  # 转置: [seq_length + 1, batch_size, self.width]

            # 通过对应的 transformer 堆叠
            x = self.transformers_tokens[i](x)

            x = x.permute(1, 0, 2)  # 转回原始形状: [batch_size, seq_length + 1, self.width]

            # 提取最后一个位置的 token
            token_i = x[:, -1, :]  # [batch_size, self.width]
            prev_tokens.append(token_i.unsqueeze(1))  # 存储当前 token

            # 更新 x_info，将当前 token 拼接到 x_info
            x_info = torch.cat([x_info, token_i.unsqueeze(1)], dim=1)

        # 拼接所有生成的 tokens
        # [b, 256(s), 256(c)]
        tokens = torch.cat(prev_tokens, dim=1)  
        tokens = self.ln_post(tokens)  # 归一化

        # [b, 256(c), 256(s)]

 
        tokens=tokens.reshape(tokens.shape[0], tokens.shape[1], 16, 16)
        # [b,256(s),16(c),16(c)]
        return tokens



    

