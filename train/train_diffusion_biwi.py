import os
import sys

sys.path.append(".")
import torch
from tqdm import tqdm
from video_diffusion_pytorch.diffusion_BIWI_encoder_decoder import GaussianDiffusion
from models.fdm import FDM
from datasets.data_loader import get_dataloaders

import numpy as np
from models.vq_vae import VQAutoEncoder

from torch.utils.tensorboard import SummaryWriter
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import torch2mesh

from models.utils.config import biwi_vq_vae_args
   
import warnings
warnings.filterwarnings('ignore')

def main():
    vq_args = biwi_vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('/data/WX/fdm/checkpoints_code_error/vq_vae/biwi_stage1.pth.tar')['state_dict'])
    freeze(autoencoder)  # 冻结vqvae模型

    model = FDM(feature_dim=1024)

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2 or cross
    )

    load_model = False
    dev = 'cuda:1'

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    train_loader = loader['train']

    train_epoch = 50
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)

    save_path = './checkpoints/diffusion_BIWI_origin_vqvae'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    if load_model:
        print('load pretrained model from checkpoints')
        load('./checkpoints/diffusion_BIWI_origin_vqvae', '100', diffusion, optimizer)

    diffusion.train()
    autoencoder.eval()
    diffusion.to(dev)
    autoencoder.to(dev)

    for epoch in range(train_epoch):
        epoch_log = epoch + 1
        print(f'Starting epoch {epoch_log}')

        loss = run_step(train_epoch, epoch_log, optimizer, train_loader, diffusion, autoencoder, writer, save_path, dev)
        print(f'Epoch {epoch_log} loss: {loss}')
        writer.add_scalar('Loss/train_epoch', loss, epoch_log)

def run_step(epochs, epoch_log, optimizer, train_loader, diffusion, autoencoder, writer, save_path, dev):
    sum_loss = 0
    with tqdm(range(len(train_loader)), desc=f'Train[{epoch_log}/{epochs}]') as tbar:
        for i, (audio, motion, template, id_one_hot, file_name) in enumerate(train_loader):
            optimizer.zero_grad()

            audio = audio.to(dev)
            motion = motion.to(dev)
            template = template.to(dev)
            id_one_hot = id_one_hot.to(dev)

            latent_motion = autoencoder.encode(motion - template)
            denoise_loss, result = diffusion(latent_motion, audio, id_one_hot)
            quanted, _, _ = autoencoder.quant(result)
            output_motion = autoencoder.decode(quanted) + template


            loss_recon = recone_loss(output_motion, motion)
            # q_recon = recone_loss(feat_q_gt, feat_out_q)

            # tbar.set_postfix(vq_loss=vq_loss.item(), noise_loss=denoise_loss.item(), loss_recon=loss_recon.item(), loss=loss.item())  # 为tqdm添加loss项
            tbar.set_postfix(loss_recon=loss_recon.item(), noise_loss=denoise_loss.item())
            tbar.update(1)

            loss = denoise_loss + loss_recon
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            writer.add_scalar('Loss/denoise', denoise_loss, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/vq_loss', vq_loss, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/recon', loss_recon, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/train', loss, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/face_move', move_recone, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/face_move_origin', move_origin, (epoch_log - 1) * len(train_loader) + i)

    if epoch_log % 5 == 0:
        save(save_path, epoch_log, diffusion, optimizer)
    
    return sum_loss / len(train_loader)

# 将motion与audio的长度统一
def align_motion_audio(all_audio, all_motion):
    if all_audio.shape[1] // 2 > all_motion.shape[1]:
        all_audio = all_audio[:, :all_motion.shape[1]]
    elif all_audio.shape[1] // 2 < all_motion.shape[1]:
        all_motion = all_motion[:, :all_audio.shape[1] // 2]
    return all_audio, all_motion

def vq_vae_loss(output, target):
    loss = torch.nn.functional.mse_loss(output, target)
    return loss

def recone_loss(output_motion, motion):
    if output_motion.shape[1] != motion.shape[1]:
        motion = motion[:, :output_motion.shape[1], :]
    loss = torch.nn.functional.mse_loss(output_motion, motion)
    # pdist = torch.nn.PairwiseDistance(p=2)
    # output = pdist(output_motion, motion)
    # loss = torch.mean(output)
    return loss

def save(save_path, epoch, model, opt):
        data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }
        torch.save(data, str(save_path + f'/model-{epoch}.mpt'))


def load(save_path, epoch, model, opt):
    data = torch.load(str(save_path + f'/model-{epoch}.mpt'))
    model.load_state_dict(data['model'])
    # opt.load_state_dict(data['opt'])

# 加载Motion Encoder和Motion Decoder
def load_encoder_decoder(load_path, epoch, encoder, decoder):
    print(f'load encoder decoder checkpoint from {load_path}/model-{epoch}.mpt')
    checkpoint = torch.load(str(load_path + f'/model-{epoch}.mpt'))['model']
    
    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k[15:]
        weights_dict[new_k] = v
        
    encoder_dict = encoder.state_dict()
    decoder_dict = decoder.state_dict()

    enc_state_dict = {k:v for k,v in weights_dict.items() if k in encoder_dict.keys()}
    dec_state_dict = {k:v for k,v in weights_dict.items() if k in decoder_dict.keys()}
    # dec_state_dict = {k:v for k,v in checkpoint.items() if k in denoise.keys()}
    print('load encoder decoder')
    encoder.load_state_dict(enc_state_dict, strict=False)
    decoder.load_state_dict(dec_state_dict, strict=False)

    freeze(encoder)
    freeze(decoder)

# 冻结某一模型
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

if __name__ == "__main__":
    main()