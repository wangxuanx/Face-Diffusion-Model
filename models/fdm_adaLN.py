import math
import numpy as np
import torch
import torch.nn as nn


class FDM(nn.Module):
    def __init__(self, feature_dim=1024, vertice_dim=70110, n_head=4, num_layers=8, struct='Dec'):
        super(FDM, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.struct = struct
        self.audio_feature_map = nn.Linear(768, feature_dim)
        # time embedding
        self.time_embedd = TimestepEmbedder(feature_dim, PositionalEncoding(feature_dim, 0.1))
        # positional encoding 
        self.PE = PositionalEncoding(feature_dim)
        # temporal bias
        if struct == 'Enc':
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif struct == 'Dec':
            decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)        
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, audio, t, vertice, past_motion):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.

        audio_feature = self.audio_feature_map(audio)

        time = self.time_embedd(t.unsqueeze(1))

        tens_input = torch.cat((time, audio_feature, past_motion, vertice), 1).transpose(0, 1)

        if self.struct == 'Enc':
            # tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            tens_input = self.PE(tens_input)
            feat_out = self.transformer_encoder(tens_input)
            feat_out = feat_out[-1:, :, :].transpose(0, 1)
        elif self.struct == 'Dec':
            tgt_mask = self.biased_mask[:, :vertice.shape[1], :vertice.shape[1]].clone().detach().to(device=self.device)
            memory_mask = self.enc_dec_mask(self.device, self.dataset, vertice.shape[1], tens_input.shape[1])
            feat_out = self.transformer_decoder(vertice, tens_input, tgt_mask=tgt_mask, memory_mask=memory_mask)

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