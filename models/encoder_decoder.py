import torch
import torch.nn as nn

# 预训练的一个Motion Encoder和Motion Decoder

class Motion_Encoder(nn.Module):
    def __init__(self, latent_dim, motion_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_dim = motion_dim

        self.motion_encoder = nn.Sequential(
            nn.Linear(self.motion_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, motion):
        motion = self.motion_encoder(motion)
        return motion
    
class Motion_Decoder(nn.Module):
    def __init__(self, latent_dim, motion_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_dim = motion_dim

        self.motion_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.motion_dim),
        )

    def forward(self, motion):
        motion = self.motion_decoder(motion)
        return motion
    
class Trainer(nn.Module):
    def __init__(self, latent_dim, motion_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_dim = motion_dim

        self.motion_encoder = Motion_Encoder(latent_dim, motion_dim)
        self.motion_decoder = Motion_Decoder(latent_dim, motion_dim)

    def forward(self, motion, template):
        motion = motion - template  # 将template减去只获得motion
        motion = self.motion_encoder(motion)
        motion = self.motion_decoder(motion)
        motion = motion + template  # 将template加回来
        return motion

    def loss_fun(self, motion, motion_recon):
        loss_l2 = nn.L1Loss()(motion, motion_recon)
        return loss_l2