import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, level=2.5):
        super().__init__()
        self.model = model  # model is the actual model to ru
        self.level = level

    def forward(self, audio, t, x_noisy, one_hot):

        out = self.model(audio, t, x_noisy, one_hot, False, train=False)
        out_uncond = self.model(audio, t, x_noisy, one_hot, True, train=False)

        scale = torch.ones(x_noisy.shape[1], device=audio.device) * self.level
        return out_uncond + (scale.view(1, -1, 1) * (out - out_uncond))
