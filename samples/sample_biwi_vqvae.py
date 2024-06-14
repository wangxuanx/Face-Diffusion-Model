import argparse
import sys

import numpy as np
sys.path.append(".")
import os
import torch
from datasets.data_loader import get_dataloaders
from models.vq_vae import VQAutoEncoder

from models.utils.config import biwi_vq_vae_args

def main():


    vq_args = biwi_vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('checkpoints/vqvae_BIWI/model-100.mpt')['model'])

    save_path = 'checkpoints/vqvae_BIWI/result'
    dev = 'cuda:1'
    autoencoder.eval()  # 评估vqvae模型
    autoencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=False)
    test_loader = loader['test']

    # 对数据进行采样
    sample_step(test_loader, dev, autoencoder, save_path)


def sample_step(test_loader, dev, autoencoder, save_folder):
    for n, (motion, template, one_hot, file_name) in enumerate(test_loader):

        motion = motion.to(dev)
        template = template.to(dev)
        one_hot = one_hot.to(dev)

        h = autoencoder.encode(motion - template)
        quanted, _, _ = autoencoder.quant(h)
        output_motion = autoencoder.decode(quanted) + template
        
        # feature decoding
        output_motion = output_motion.cpu().detach().numpy()

        np.save(os.path.join(save_folder, file_name[0][:-4]), output_motion)

if __name__ == "__main__":
    main()