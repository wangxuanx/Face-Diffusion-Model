import argparse
import sys

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm
from datasets.data_loader_mead import get_dataloaders
from models.vq_vae_emotion import VQAutoEncoder
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import get_mesh, torch2mesh

def main():
    flame_config = get_config()
    flame = FLAME(flame_config)  # 加载FLAME模型

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vqvae_mead/model-30.mpt')['model'])

    save_path = './checkpoints/vqvae_mead/result_30'
    dev = 'cuda:1'
    autoencoder.eval()  # 评估vqvae模型
    autoencoder.to(dev)
    flame.eval()
    flame.to(dev)
    

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10)
    test_loader = loader['test']

    # 对数据进行采样
    sample_step(test_loader, dev, flame, autoencoder, save_path)


def sample_step(test_loader, dev, flame, autoencoder, save_folder):
    for n, (mead_motion, mead_template, one_hot, id_one_hot, file_name) in enumerate(test_loader):
        mead_motion = mead_motion.to(dev)
        mead_template = mead_template.to(dev)
        one_hot = one_hot.to(dev)

        motion = torch2mesh(flame, mead_motion[:, :, :50], mead_motion[:, :, 50:])
        template = torch2mesh(flame, mead_template[:, :, :50], mead_template[:, :, 50:])

        # motion = motion.unsqueeze(0)

        h = autoencoder.encode(motion - template, one_hot)
        quanted, _, _ = autoencoder.quant(h, one_hot)
        output_motion = autoencoder.decode(quanted) + template
        
        # feature decoding
        output_motion = output_motion.cpu().detach().numpy()

        np.save(os.path.join(save_folder, file_name[0][:-4]), output_motion)

def vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256 * 7, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=64, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=5023*3, help='number of input channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units')
    parser.add_argument('--neg', type=int, default=0.2, help='number of negative samples')
    parser.add_argument('--quant_factor', type=float, default=0, help='number of quantization factor')
    parser.add_argument('--INaffine', type=bool, default=False, help='number of INaffine')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1536, help='number of intermediate size')
    parser.add_argument('--face_quan_num', type=int, default=8, help='number of face quantization')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()