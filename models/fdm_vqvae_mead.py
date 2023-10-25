import math
import numpy as np
import torch
import torch.nn as nn
from models.hubert import HubertModel
from utiles.adaIN import adaptive_instance_normalization as adaIN

class FDM(nn.Module):
    def __init__(self, feature_dim=512, vertice_dim=70110, n_head=4, num_layers=8, struct='Enc'):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.struct = struct
        self.audio_encoder = HubertModel.from_pretrained('/data/WX/hubert-large-ls960-ft')
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_extract = nn.Sequential(
            nn.Linear(1024 * 2, feature_dim),  # 128 1024
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim)
        )
    
        # time embedding
        self.one_hot_timesteps = np.eye(1000)
        # self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        self.time_embedd = nn.Sequential(
            nn.Linear(1000, feature_dim),
            nn.Mish(),
        )
        # motion embedding
        self.emotion_embedd = nn.Linear(7, feature_dim)
        # style embedding
        self.style_embedd = nn.Linear(25, feature_dim)
        # latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Mish()
        )
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # latent decoder
        self.latent_decoder = nn.Linear(feature_dim, feature_dim)
        nn.init.constant_(self.latent_decoder.weight, 0)
        nn.init.constant_(self.latent_decoder.bias, 0)

    def mask_cond(self, cond, train=False, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif train:
            mask = torch.bernoulli(torch.ones_like(cond, device=cond.device) * 0.1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, audio, t, vertice, emotion_one_hot, id_one_hot, mask_cond=False, train=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        device = audio.device
        
        audio = self.audio_encoder(audio).last_hidden_state
        B, N, M = audio.shape
        
        audio = audio.reshape(B, N // 2, M * 2)  # (B, N // 2, M * 2)
        vertice = vertice.reshape(B, vertice.shape[1] // 8, vertice.shape[2] * 8)  # (B, N // 2, V * 3)
        num_frames = min(audio.shape[1], vertice.shape[1])
        audio = audio[:, :num_frames, :]
        vertice = vertice[:, :num_frames, :]

        audio_feature = self.audio_extract(audio)
        vertice_feature = self.latent_encoder(vertice)

        times = torch.FloatTensor(self.one_hot_timesteps[t]).to(device)
        time = self.time_embedd(times)

        emotion = self.emotion_embedd(emotion_one_hot).unsqueeze(0)

        # # style embedding
        style = self.style_embedd(id_one_hot).unsqueeze(0)
        
        vertice_feature = vertice_feature + style + emotion
        
        audio_feature += time
        # vertice_feature += emotion
        # audio_feature = adaIN(audio_feature, time.repeat(1, N // 2, 1))  # 使用adaIN对audio_feature添加时间编码
        # vertice_feature = adaIN(vertice_feature, emotion.repeat(1, N // 2, 1))  # 使用adaIN对vertice添加表情编码

        vertice_feature = self.PE(vertice_feature)
        tgt_mask = self.biased_mask[:, :vertice_feature.shape[1], :vertice_feature.shape[1]].clone().detach().to(device=device)
        memory_mask = enc_dec_mask(device, 'BIWI', vertice_feature.shape[1], audio_feature.shape[1])
        feat_out = self.transformer_decoder(vertice_feature, audio_feature, tgt_mask=tgt_mask, memory_mask=memory_mask)

        feat_out = self.latent_decoder(feat_out)
        feat_out = feat_out.reshape(B, feat_out.shape[1] * 8, feat_out.shape[2] // 8)
        return feat_out

    def predict(self, audio, t, vertice, one_hot):

        B, N, M = audio.shape

        audio_feature = self.audio_extract(audio)
        time = self.time_embedd(t.unsqueeze(1))
        vertice = vertice.flatten(-2).unsqueeze(1)
        
        # style embedding
        style = self.learnable_style_emb(one_hot).unsqueeze(0).repeat(B, 1, 1)

        tens_input = torch.cat((time, audio_feature, style, vertice), 1).permute(1, 0, 2)

        if self.struct == 'Enc':
            tens_input = self.PE(tens_input)
            feat_out = self.transformer_encoder(tens_input)
            feat_out = feat_out[-1:, :, :].transpose(0, 1)
        elif self.struct == 'Dec':
            tgt_mask = self.biased_mask[:, :vertice.shape[1], :vertice.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice.shape[1], tens_input.shape[1])
            feat_out = self.transformer_decoder(vertice, tens_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        feat_out = self.motion_decoder(feat_out).reshape(B, -1, 128)
        return feat_out

# Temporal Bias
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
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
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
            mask[i, i] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
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
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)