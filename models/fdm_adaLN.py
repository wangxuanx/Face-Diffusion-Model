import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).flatten(-2).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FDM(nn.Module):
    def __init__(self, feature_dim=1024, vertice_dim=70110, n_head=16, num_layers=24):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.audio_feature_map = nn.Linear(768, feature_dim)
        # time embedding
        self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        # style embedding
        self.learnable_style_emb = nn.Linear(len('F2 F3 F4 M3 M4 M5 F1 F5 F6 M1'.split(' ')), feature_dim)
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        self.blocks = nn.ModuleList([
            DiTBlock(feature_dim, n_head, mlp_ratio=4.0) for _ in range(num_layers)
        ])
        # motion decoder
        self.motion_decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim, bias=False)
        )

    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, audio, t, vertice, one_hot):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        B, T, V = vertice.shape

        audio_feature = self.audio_feature_map(audio)
        time = self.time_embedd(t.unsqueeze(1))
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0)
        
        cond = time + audio_feature
        vertice = vertice.flatten(-2).unsqueeze(1) + style

        for block in self.blocks:
            x = block(vertice, cond)

        x = self.motion_decoder(x)
        x = x.reshape(B, T, V)
        return x

    def predict(self, audio, template, one_hot, one_hot2=None, weight_of_one_hot=None):
        template = template.unsqueeze(1) # (1,1, V*3)

        # style embedding
        obj_embedding = self.learnable_style_emb(torch.argmax(one_hot, dim=1))

        # style interpolation (optional)
        if one_hot2 is not None and weight_of_one_hot is not None:
            obj_embedding2 = self.learnable_style_emb(torch.argmax(one_hot2, dim=1))
            obj_embedding = obj_embedding * weight_of_one_hot + obj_embedding2 * (1-weight_of_one_hot)
        obj_embedding = obj_embedding.unsqueeze(1)

        # audio feature extraction
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        # autoregressive facial motion prediction 
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = self.enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            feat_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            feat_out = self.feat_map(feat_out)

            feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1]*self.face_quan_num, -1)
            # predicted feature to quantized one
            feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
            # quantized feature to vertice
            if i == 0:
                vertice_out_q = self.autoencoder.decode(torch.cat([feat_out_q, feat_out_q], dim=-1))
                vertice_out_q = vertice_out_q[:,0].unsqueeze(1)
            else:
                vertice_out_q = self.autoencoder.decode(feat_out_q)

            if i != frame_num - 1:
                new_output = self.vertice_map(vertice_out_q[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)


        # quantization and decoding
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
        vertice_out = self.autoencoder.decode(feat_out_q)

        vertice_out = vertice_out + template
        return vertice_out
    
    def enc_dec_mask(self, device, T, S):
        mask = torch.ones(T, S)
        
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
        
        return (mask==1).to(device=device)

# 时间编码
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        timesteps = self.sequence_pos_encoder.pe[timesteps]
        return self.time_embed(timesteps).squeeze(1)


# 添加位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)