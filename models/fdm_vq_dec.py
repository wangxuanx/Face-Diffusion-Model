import math
import numpy as np
import torch
import torch.nn as nn
from zmq import device
from models.wav2vec import Wav2Vec2Model


class FDM(nn.Module):
    def __init__(self, feature_dim=512, vertice_dim=70110, n_head=16, num_layers=18):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.audio_extract = nn.Sequential(
            nn.Linear(768, feature_dim),  # 128 1024
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim, bias=False)
        )
        # time embedding
        self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        # style embedding
        self.learnable_style_emb = nn.Linear(len('F2 F3 F4 M3 M4 M5 F1 F5 F6 M1'.split(' ')), feature_dim)
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        # self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = 600, period=25)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # motion decoder
        self.motion_decoder = nn.Linear(feature_dim, feature_dim, bias=False)

        # 对模型中的Linear进行初始化
        nn.init.constant_(self.audio_extract[2].weight, 0)
        nn.init.constant_(self.motion_decoder.weight, 0)


    def forward(self, audio, t, vertice, past_motion, one_hot):
        dev = audio.device
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        B, N, M = audio.shape

        audio_feature = self.audio_extract(audio)
        time = self.time_embedd(t.unsqueeze(1))

        audio_feature += time  # 添加时间编码
        
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0).repeat(B, 1, 1)
        vertice = torch.cat((past_motion, vertice), 2).transpose(1, 2)
        vertice += style  # 添加风格编码
        
        feat_out = self.transformer_decoder(vertice, audio_feature)

        feat_out = self.motion_decoder(feat_out)
        feat_out = feat_out.transpose(1, 2)
        return feat_out

    def predict(self, audio, t, vertice, past_motion, one_hot):

        B, N, M = audio.shape

        audio_feature = self.audio_extract(audio)
        time = self.time_embedd(t.unsqueeze(1))

        audio_feature += time  # 添加时间编码
        
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0).repeat(B, 1, 1)
        vertice = torch.cat((past_motion, vertice), 2).transpose(1, 2)
        vertice += style  # 添加风格编码
        
        feat_out = self.transformer_decoder(vertice, audio_feature)

        feat_out = self.motion_decoder(feat_out)
        feat_out = feat_out.transpose(1, 2)
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
    
# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)