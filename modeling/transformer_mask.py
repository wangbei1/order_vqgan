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



def create_attention_mask(L, device):
    attn_mask = torch.zeros(L, L, device=device, dtype=torch.bool)
    causal_mask = torch.triu(torch.ones(L , L , device=device, dtype=torch.bool), diagonal=1)
    return causal_mask



class Transformer_mask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.width = config.transformer.width
        self.num_layers = config.transformer.num_layers
        self.num_heads = config.transformer.num_heads
        scale = self.width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.layerNorm = nn.LayerNorm(self.width)
        self.ln_pre = nn.LayerNorm(self.width)  

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = pixel_values

        # [b,256(c),256(s)]
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # [b,256(s),256(c)]
        x = x.permute(0, 2, 1) 

        # [b,257(s),256(c)]
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)


        x = self.ln_pre(x)

        # [257(s),b,256(c)]
        x = x.permute(1, 0, 2)  # NLD -> LND, x shape [L, B, d_model], L=总序列长度

        L = x.shape[0]  
        attn_mask = create_attention_mask(L, x.device)

        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask=attn_mask)

        # [b,257(s),256(c)]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.layerNorm(x)

        # [b,256(s),256(c)]
        x = x[:, 1:,:]



        x=x.reshape(x.shape[0], x.shape[1], 16, 16)
        
        # [b,256(s),16(c),16(c)]
        return x  
    

