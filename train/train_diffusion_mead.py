import os
import sys

sys.path.append(".")
import torch
from tqdm import tqdm
from video_diffusion_pytorch.diffusion_mead_encoder_decoder import GaussianDiffusion
from models.fdm_vqvae_mead import FDM
from datasets.data_loader_mead import get_dataloaders

import numpy as np
from models.vq_vae_emotion import VQAutoEncoder

from torch.utils.tensorboard import SummaryWriter
from utiles.args import vq_vae_args
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import torch2mesh

from models.utils.config import vq_vae_args

import warnings
warnings.filterwarnings('ignore')

def main():
    flame_config = get_config()
    flame = FLAME(flame_config)  # 加载FLAME模型

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vqvae_mead/model-30.mpt')['model'])
    freeze(autoencoder)  # 冻结vqvae模型

    model = FDM(feature_dim=512, vertice_dim=5023 * 3, struct='Dec')

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2 or cross
    )

    load_model = False
    dev = 'cuda:1'

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    train_loader = loader['train']

    train_epoch = 100
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)

    save_path = './checkpoints/diffusion_mead_vqvae'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    if load_model:
        print('load pretrained model from checkpoints')
        load('./checkpoints/diffusion_mead_encoder_decoder', '100', diffusion, optimizer)

    diffusion.train()
    flame.eval()
    autoencoder.eval()
    diffusion.to(dev)
    autoencoder.to(dev)
    flame.to(dev)

    for epoch in range(train_epoch):
        epoch_log = epoch + 1
        print(f'Starting epoch {epoch_log}')

        loss = run_step(train_epoch, epoch_log, optimizer, train_loader, flame, diffusion, autoencoder, writer, save_path, dev)
        print(f'Epoch {epoch_log} loss: {loss}')
        writer.add_scalar('Loss/train_epoch', loss, epoch_log)

def run_step(epochs, epoch_log, optimizer, train_loader, flame, diffusion, autoencoder, writer, save_path, dev):
    sum_loss = 0
    with tqdm(range(len(train_loader)), desc=f'Train[{epoch_log}/{epochs}]') as tbar:
        for i, (audio, mead_motion, mead_template, emotion_one_hot, id_one_hot, file_name) in enumerate(train_loader):
            optimizer.zero_grad()

            audio = audio.to(dev)
            mead_motion = mead_motion.to(dev)
            mead_template = mead_template.to(dev)
            emotion_one_hot = emotion_one_hot.to(dev)
            id_one_hot = id_one_hot.to(dev)

            motion = torch2mesh(flame, mead_motion[:, :, :50], mead_motion[:, :, 50:])  # frame*5023*3
            template = torch2mesh(flame, mead_template[:, :, :50], mead_template[:, :, 50:])

            latent_motion = autoencoder.encode(motion - template, emotion_one_hot)
            denoise_loss, result = diffusion(latent_motion, audio, emotion_one_hot, id_one_hot)
            quanted, _, _ = autoencoder.quant(result, emotion_one_hot)
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

def get_audio_motion_signal(all_audio, all_motion,  batchsize, num_pre=0):
    num_frames = all_audio.shape[1] // 2

    pad_audio_idx = torch.tensor([0, 1] * num_pre + [i for i in range(0, num_frames * 2)] + [num_frames * 2 - 2, num_frames * 2 - 1] * num_pre)
    pad_motion_idx = torch.tensor([0] * num_pre + [i for i in range(0, num_frames)] + [num_frames - 1] * num_pre)
    all_audio = all_audio[:, pad_audio_idx, :]
    all_motion = all_motion[:, pad_motion_idx, :]
                                   
    random_idx = torch.randint(num_frames, (batchsize,))
    random_idx += num_pre

    sliced_audio = [] 
    sliced_motion = []

    for index in random_idx:

        sliced_tensor = all_audio[:, (index - num_pre) * 2:(index + num_pre + 1) * 2, :]
        sliced_audio.append(sliced_tensor)
        sliced_tensor = all_motion[:, index:index + 1, :]
        sliced_motion.append(sliced_tensor)

    audio = torch.stack(sliced_audio).squeeze(1)
    motion = torch.stack(sliced_motion).squeeze(1)

    return audio, motion


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