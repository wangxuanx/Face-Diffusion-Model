import math
import numpy as np
import torch
import torch.nn as nn
from models.wav2vec import Wav2Vec2Model
from utiles.adaIN import adaptive_instance_normalization as adaIN


class FDM(nn.Module):
    def __init__(self, feature_dim=1024, vertice_dim=70110, n_head=4, num_layers=8, struct='Enc'):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.struct = struct
        self.audio_encoder = Wav2Vec2Model.from_pretrained('/data/WX/wav2vec2-base-960h')
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_extract = nn.Sequential(
            nn.Linear(768 * 2, feature_dim),  # 128 1024
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim)
        )
        # time embedding
        self.one_hot_timesteps = np.eye(1000)  # 1000步的步数
        # self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        self.time_embedd = nn.Sequential(
            nn.Linear(1000, feature_dim),
            nn.Mish(),
        )
        # style embedding
        self.style_embedd = nn.Sequential(
            nn.Linear(6, feature_dim),
            nn.Mish(),
        )
        self.latent_encoder = nn.Linear(feature_dim, feature_dim)
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        if struct == 'Enc':
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif struct == 'Dec':
            self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = 600, period=25)
            decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)        
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.latent_decoder = nn.Linear(feature_dim, feature_dim)
        nn.init.constant_(self.latent_decoder.weight, 0)
        nn.init.constant_(self.latent_decoder.bias, 0)

    def mask_cond(self, cond, train=False, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif train:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * 0.1).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, audio, t, vertice, one_hot):
        device = audio.device
        
        audio = self.audio_encoder(audio).last_hidden_state
        B, N, M = audio.shape
        audio = audio.reshape(B, N // 2, M * 2)  # (B, N // 2, M * 2)
        num_frames = min(audio.shape[1], vertice.shape[1])
        audio = audio[:, :num_frames, :]
        vertice = vertice[:, :num_frames, :]

        audio_feature = self.audio_extract(audio)
        vertice_feature = self.latent_encoder(vertice)

        times = torch.FloatTensor(self.one_hot_timesteps[t]).to(device)
        time = self.time_embedd(times).reshape(-1, 1, 1024)

        style = self.style_embedd(one_hot)
        vertice_feature = vertice_feature + style

        if self.struct == 'Enc':
            # tens_input = torch.cat((time, audio_feature, past_motion, vertice), 1).transpose(0, 1)
            tens_input = torch.cat((time, audio_feature, vertice_feature), 1).transpose(0, 1)
            # tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tens_input = self.PE(tens_input)
            feat_out = self.transformer_encoder(tens_input)
            feat_out = feat_out[-1:, :, :].transpose(0, 1)
        elif self.struct == 'Dec':
            audio_feature += time
            vertice_feature = self.PE(vertice_feature)
            tgt_mask = self.biased_mask[:, :vertice_feature.shape[1], :vertice_feature.shape[1]].clone().detach().to(device)
            memory_mask = enc_dec_mask(device, 'BIWI', vertice_feature.shape[1], audio_feature.shape[1])
            feat_out = self.transformer_decoder(vertice_feature, audio_feature, tgt_mask=tgt_mask, memory_mask=memory_mask)

        feat_out = self.latent_decoder(feat_out)
        return feat_out

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
        x = x.transpose(0, 1)
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        x = x.transpose(0, 1)
        return self.dropout(x)