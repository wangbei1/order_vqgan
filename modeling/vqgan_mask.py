from .maskgit_blocks import Decoder
from .maskgit_blocks import Encoder 
from .maskgit_blocks import VectorQuantizer
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from .transformer_mask import Transformer_mask
from .transformer_causual import Transformer_causal
from modeling.utils.utils import instantiate_from_config

class VQGAN_mask(nn.Module):
    def __init__(self, config):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Encoder(conf)
        self.decoder = Decoder(conf)
        self.quantize = VectorQuantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # pretrained_weight=config.model.pretrained_weight
        # if pretrained_weight is not None:
        #     pretrained_dict = torch.load(pretrained_weight, map_location=torch.device("cpu"))
            
        #     encoder_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('encoder')}
            
        #     self.encoder.load_state_dict(encoder_dict, strict=False)
        
        lossconfig = config.model.lossconfig
        self.loss = instantiate_from_config(lossconfig)    
        self.transformer=Transformer_mask(config)
    def encode(self, x):
        hidden_states = self.encoder(x)
        
        hidden_states=self.transformer(hidden_states)
        # hidden_states.shape=[B,256(s),16(c),16(c)]
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)

        # [B,256(s),256(c)]
        quantized_states = quantized_states.reshape(quantized_states.shape[0], quantized_states.shape[1], -1)
        # [b,256(c),256(s)]
        quantized_states = quantized_states.permute(0, 2, 1)  
        
        # quantized_states.shape=[B,256(c),16(s),16(s)]
        quantized_states = quantized_states.reshape(quantized_states.shape[0], quantized_states.shape[1], 16, 16)

        return quantized_states,codebook_indices.detach(),codebook_loss
    
    def decode(self, quantized_states):
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    def decode_tokens(self, codebook_indices):
        quantized_states = self.quantize.get_codebook_entry(codebook_indices)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def remove_module_prefix(self,state_dict):
        """Removes the 'module.' prefix from state_dict keys."""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        return new_state_dict

    def init_from_ckpt(self, path_decoder,ignore_keys=list()):
        checkpoint = torch.load(path_decoder, map_location="cpu")
        checkpoint_VQ=checkpoint["model"]
        new_state_dict=self.remove_module_prefix(checkpoint_VQ)
        self.load_state_dict(new_state_dict) 
        print(f"Restored from {path_decoder}")