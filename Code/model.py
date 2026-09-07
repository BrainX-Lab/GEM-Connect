import BNT_modules as modules
from moe import MyMoE
import torch
from torch import nn

class Brain_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.CLIP.num_heads
        self.encoder_layer_sizes = [config.CLIP.n_roi] + config.CLIP.encoder_layer_sizes
        self.decoder_layer_sizes = config.CLIP.decoder_layer_sizes + [config.CLIP.n_roi]
        
        # SC to embed
        self.SC_encoder = modules.Transformer(
            layer_sizes=self.encoder_layer_sizes,
            heads=self.num_heads,
            batch_first=True
        )
        
        # FC to embed
        self.FC_encoder = modules.Transformer(
            layer_sizes=self.encoder_layer_sizes,
            heads=self.num_heads,
            batch_first=True
        )
        
        # embed to SC
        self.SC_decoder = modules.Transformer(
            layer_sizes=self.decoder_layer_sizes,
            heads=self.num_heads,
            batch_first=True
        )
        
        # embed to FC
        self.FC_decoder = modules.Transformer(
            layer_sizes=self.decoder_layer_sizes,
            heads=self.num_heads,
            batch_first=True
        )
        
        # genetic MoE
        self.SC_moe = MyMoE(config.MOE)
        self.FC_moe = MyMoE(config.MOE)
        
        self.input_type = self.SC_encoder.encoder_layers[0].self_attn.in_proj_weight.dtype
    
    def encode_SC(self, image:torch.Tensor):
        return self.SC_encoder(image.type(self.input_type))
    
    def encode_FC(self, image:torch.Tensor):
        return self.FC_encoder(image.type(self.input_type))
    
    def decode_SC(self, embed:torch.Tensor, genetic:torch.Tensor):
        moe_out, topk_idx = self.SC_moe(embed.type(self.input_type), genetic)
        return self.SC_decoder(moe_out), topk_idx
    
    def decode_FC(self, embed:torch.Tensor, genetic:torch.Tensor):
        moe_out, topk_idx = self.FC_moe(embed.type(self.input_type), genetic)
        return self.FC_decoder(moe_out), topk_idx
    
    # encoder & decoder for either FC or SC
    def forward(self, batch_SC:torch.Tensor, batch_FC:torch.Tensor, batch_genetic:torch.Tensor):
        # encoders                                                     input: [batch_size, ROI, ROI]
        embed_SC = self.encode_SC(batch_SC)                                 # [batch_size, ROI, embed_dim]
        embed_FC = self.encode_FC(batch_FC)
        
        # moe + decoders                                               input: [batch_size, ROI, embed_dim] 
        pred_SC, _ = self.decode_SC(embed_FC, batch_genetic)                # [batch_size, ROI, ROI]
        pred_FC, _ = self.decode_FC(embed_SC, batch_genetic)
        
        # regeneration
        pred_SC_t = pred_SC.transpose(-2, -1)
        pred_SC_full = (pred_SC + pred_SC_t) / 2
        pred_FC_t = pred_FC.transpose(-2, -1)
        pred_FC_full = (pred_FC + pred_FC_t) / 2
        return pred_SC_full, pred_FC_full, embed_SC, embed_FC
    
    def predict_FC(self, SC_data:torch.Tensor, genetic_data:torch.Tensor):
        embed = self.encode_SC(SC_data)
        pred, topk_idx = self.decode_FC(embed, genetic_data)
        pred_t = pred.transpose(-2, -1)
        pred_full = (pred + pred_t) / 2
        return pred_full, topk_idx
        
    def predict_SC(self, FC_data:torch.Tensor, genetic_data:torch.Tensor):
        embed = self.encode_FC(FC_data)
        pred, topk_idx = self.decode_SC(embed, genetic_data)
        pred_t = pred.transpose(-2, -1)
        pred_full = (pred + pred_t) / 2
        return pred_full, topk_idx
 
        
