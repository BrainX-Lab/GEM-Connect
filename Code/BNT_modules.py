from torch.nn import TransformerEncoderLayer
import torch
from torch import Tensor, nn
from typing import Optional
import torch.nn.functional as F

# used as a single layer for the transformer block
### Source code from BNT https://github.com/Wayfear/BrainNetworkTransformer ###
class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=4, 
                 dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, 
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor],
                  is_causal=False) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)
### End Source Code ###

# transformer block with multiple layers of difference input lengths
class Transformer(nn.Module):
    def __init__(self, layer_sizes, heads, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        super().__init__()
        self.num_layers = len(layer_sizes)
        self.encoder_layers = nn.ModuleList([InterpretableTransformerEncoder(d_model=x, nhead=heads, dim_feedforward=dim_feedforward, 
                                                                             dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, 
                                                                             batch_first=batch_first, norm_first=norm_first, bias=bias, device=device, dtype=dtype)
                                             for x in layer_sizes[:-1]])
        self.proj_layers    = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1], device=device, dtype=dtype) 
                                             for i in range(self.num_layers - 1)])       
    
    def forward(self, x:Tensor):
        for i in range(self.num_layers - 1):
            x = self.encoder_layers[i](x)
            x = self.proj_layers[i](x)          # [batch_size, seq, feature_out]
        return x 
        