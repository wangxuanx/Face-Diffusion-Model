import math
import numpy as np
import torch
import torch.nn as nn
from models.wav2vec import Wav2Vec2Model


class FDM(nn.Module):
    def __init__(self, feature_dim=512, vertice_dim=70110, n_head=16, num_layers=24, struct='Enc'):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.struct = struct
        self.audio_extract = nn.Sequential(
            nn.Linear(768, feature_dim),  # 128 1024
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim)
        )
        # time embedding
        self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        # style embedding
        self.learnable_style_emb = nn.Linear(len('F2 F3 F4 M3 M4 M5 F1 F5 F6 M1'.split(' ')), feature_dim)
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        if struct == 'Enc':
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif struct == 'Dec':
            decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)        
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # motion decoder
        self.motion_decoder = nn.Linear(feature_dim, feature_dim, bias=False)


    def forward(self, audio, t, vertice, one_hot):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        B, N, M = audio.shape

        audio_feature = self.audio_extract(audio)
        time = self.time_embedd(t.unsqueeze(1))
        # all_chunk = past_motion.chunk(2, 2)
        # past_motion = torch.cat((all_chunk[0].reshape(B, 1024, -1), all_chunk[1].reshape(B, 1024, -1)), 2)
        vertice = vertice.flatten(-2).unsqueeze(1)
        # motion = torch.cat((past_motion, vertice), 2).transpose(1, 2)
        
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0).repeat(B, 1, 1)

        # tens_input = torch.cat((time, audio_feature, past_motion, vertice), 1).transpose(0, 1)
        tens_input = torch.cat((time, audio_feature, style, vertice), 1).permute(1, 0, 2)

        if self.struct == 'Enc':
            # tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tens_input = self.PE(tens_input)
            feat_out = self.transformer_encoder(tens_input)
            feat_out = feat_out[-1:, :, :].transpose(0, 1)
        elif self.struct == 'Dec':
            tgt_mask = self.biased_mask[:, :vertice.shape[1], :vertice.shape[1]].clone().detach().to(device=self.device)
            memory_mask = self.enc_dec_mask(self.device, self.dataset, vertice.shape[1], tens_input.shape[1])
            feat_out = self.transformer_decoder(vertice, tens_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        feat_out = self.motion_decoder(feat_out).reshape(B, -1, 128)
        return feat_out

    def predict(self, audio, t, vertice, one_hot):

        B, N, M = audio.shape

        audio_feature = self.audio_extract(audio)
        time = self.time_embedd(t.unsqueeze(1))
        # all_chunk = past_motion.chunk(2, 2)
        # past_motion = torch.cat((all_chunk[0].reshape(B, 1024, -1), all_chunk[1].reshape(B, 1024, -1)), 2)
        vertice = vertice.flatten(-2).unsqueeze(1)
        # motion = torch.cat((past_motion, vertice), 2).transpose(1, 2)
        
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0).repeat(B, 1, 1)

        # tens_input = torch.cat((time, audio_feature, past_motion, vertice), 1).transpose(0, 1)
        tens_input = torch.cat((time, audio_feature, style, vertice), 1).permute(1, 0, 2)

        if self.struct == 'Enc':
            # tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tens_input = self.PE(tens_input)
            feat_out = self.transformer_encoder(tens_input)
            feat_out = feat_out[-1:, :, :].transpose(0, 1)
        elif self.struct == 'Dec':
            tgt_mask = self.biased_mask[:, :vertice.shape[1], :vertice.shape[1]].clone().detach().to(device=self.device)
            memory_mask = self.enc_dec_mask(self.device, self.dataset, vertice.shape[1], tens_input.shape[1])
            feat_out = self.transformer_decoder(vertice, tens_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        feat_out = self.motion_decoder(feat_out).reshape(B, -1, 128)
        return feat_out
    

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
            nn.Mish(),
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