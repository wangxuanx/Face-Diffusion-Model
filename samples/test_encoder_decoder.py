import sys

import numpy as np
sys.path.append('.')
import os
import torch
from zmq import device
from datasets.data_loader import get_dataloaders
from tqdm import tqdm
from models.encoder_decoder import Trainer
from torch.utils.tensorboard import SummaryWriter

# 训练一个Motion Encoder和Motion Decoder来直接采样

def dataloader(batch_size=64, workers=10, read_audio=False):
    loader = get_dataloaders(batch_size=batch_size, workers=workers, read_audio=read_audio)
    train_loader = loader['train']
    val_loader = loader['valid']
    test_loader = loader['test']

    return train_loader, val_loader, test_loader

def main():
    save_path = './checkpoints/encoder_decoder_load_model/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dev = 'cuda:1'
    load_model = True
    train_epochs = 1000

    train_loader, val_loader, test_loader = dataloader(batch_size=16, workers=10)

    model = Trainer(1024, 70110).to(dev)

    load('./checkpoints/encoder_decoder_load_model', 'best', model)

    for i, (motion, template, one_hot, file_name) in enumerate(tqdm(val_loader)):
        motion = motion.to(dev)
        template = template.to(dev)

        recon = val_step(model, motion, template)

        recon = recon.detach().cpu().numpy()
        np.save(os.path.join(save_path, file_name[0][:-4] + '.npy'), recon)


def run_step(model, motion, template):
    recon = model(motion, template)  # 重建的motion
    loss = model.loss_fun(motion, recon)
    return loss

def val_step(model, motion, template):
    recon = model(motion, template)  # 重建的motion
    return recon

def save(save_path, epoch, model, opt):
        data = {
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }
        torch.save(data, str(save_path + f'/model-{epoch}.mpt'))

def load(save_path, epoch, model):
    data = torch.load(str(save_path + f'/model-{epoch}.mpt'))
    model.load_state_dict(data['model'])



if __name__ == '__main__':
    main()