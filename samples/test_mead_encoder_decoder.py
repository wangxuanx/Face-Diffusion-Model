import sys

import numpy as np
sys.path.append('.')
import os
import torch
from zmq import device
from datasets.data_loader_mead import get_dataloaders
from tqdm import tqdm
from models.encoder_decoder import Trainer
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import get_mesh, torch2mesh

# 训练一个Motion Encoder和Motion Decoder来直接采样

def dataloader(batch_size=64, workers=10, read_audio=False):
    loader = get_dataloaders(batch_size=batch_size, workers=workers, read_audio=read_audio)
    train_loader = loader['train']
    val_loader = loader['valid']
    test_loader = loader['test']

    return train_loader, val_loader, test_loader

def main():
    dev = 'cuda:1'

    flame_config = get_config()
    flame = FLAME(flame_config).to(dev)  # 加载FLAME模型
    for param in flame.parameters():
        param.requires_grad = False

    save_path = './checkpoints/mead_encoder_decoder/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader, val_loader, test_loader = dataloader(batch_size=16, workers=10)

    model = Trainer(512, 5023 * 3).to(dev)
    flame.eval()
    model.eval()

    load('./checkpoints/mead_encoder_decoder', '400', model)

    for i, (motion, template, one_hot, file_name) in enumerate(tqdm(val_loader)):
        motion = motion.to(dev)
        template = template.to(dev)

        motion = torch2mesh(flame, motion[:, :, :50], motion[:, :, 50:])
        template = torch2mesh(flame, template[:, :, :50], template[:, :, 50:])

        recon = model(motion.flatten(-2), template.flatten(-2))  # 重建的motion

        recon = recon.detach().cpu().numpy()
        np.save(os.path.join(save_path, file_name[0][:-4] + '.npy'), recon)

def load(save_path, epoch, model):
    data = torch.load(str(save_path + f'/model-{epoch}.mpt'))
    model.load_state_dict(data['model'])



if __name__ == '__main__': 
    main()