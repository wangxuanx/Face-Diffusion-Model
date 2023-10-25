import argparse
import os
import sys

sys.path.append(".")
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from models.fdm_vq import FDM
from datasets.data_loader_mead import get_dataloaders

from models.vq_vae_emotion import VQAutoEncoder
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import get_mesh, torch2mesh

from torch.utils.tensorboard import SummaryWriter

def vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256 * 7, help='number of hidden units')  # 将结果映射在7个空间中
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

def main():
    flame_config = get_config()
    flame = FLAME(flame_config)  # 加载FLAME模型

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)

    dev = 'cuda:1'

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=False)
    train_loader = loader['train']
    val_loader = loader['valid']

    train_epoch = 400
    optimizer = torch.optim.AdamW(params=autoencoder.parameters(), lr=0.0001, amsgrad=True)

    save_path = './checkpoints/vqvae_mead'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    autoencoder.train()
    flame.eval()
    autoencoder.to(dev)
    flame.to(dev)

    for epoch in range(train_epoch):
        epoch_log = epoch + 1
        print(f'Starting epoch {epoch_log}')

        loss = run_step(train_epoch, epoch_log, optimizer, train_loader, flame, autoencoder, writer, save_path, dev)
        writer.add_scalar('Loss/train_epoch', loss, epoch_log)

        if epoch_log % 5 == 0:
            val_loss, val_recon_loss = eval_step(val_loader, flame, autoencoder, writer, save_path, dev)
            print(f'val_loss: {val_loss}, val_recon_loss: {val_recon_loss}')
            writer.add_scalar('Loss/val_epoch', val_loss, epoch_log)
            writer.add_scalar('Loss/val_recon_epoch', val_recon_loss, epoch_log)

def run_step(epochs, epoch_log, optimizer, train_loader, flame, autoencoder, writer, save_path, dev):
    
    with tqdm(range(len(train_loader)), desc=f'Train[{epoch_log}/{epochs}]') as tbar:
        sum_loss = 0
        for i, (mead_motion, mead_template, emotion_one_hot, id_one_hot, file_name) in enumerate(train_loader):
            optimizer.zero_grad()
            # F2_e03.wav ---> F2_e03.npy
            # 0 --> n
            mead_motion = mead_motion.to(dev)
            mead_template = mead_template.to(dev)
            emotion_one_hot = emotion_one_hot.to(dev)

            motion = torch2mesh(flame, mead_motion[:, :, :50], mead_motion[:, :, 50:])
            template = torch2mesh(flame, mead_template[:, :, :50], mead_template[:, :, 50:])

            recon, quant_loss, info = autoencoder(motion, template, emotion_one_hot)

            # LOSS
            loss, loss_details = vq_loss(recon, motion, quant_loss)

            tbar.set_postfix(loss_recon=loss_details[0].item(), quant_loss=loss_details[1].item(), loss=loss.item())  # 为tqdm添加loss项
            tbar.update(1)

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/recon', loss_details[0].item(), (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/all', loss.item(), (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/quant', loss_details[1].item(), (epoch_log - 1) * len(train_loader) + i)

            sum_loss += loss.item()

    if epoch_log % 5 == 0:
        save(save_path, epoch_log, autoencoder, optimizer)
    
    return sum_loss / len(train_loader)


@torch.no_grad()
def eval_step(val_loader, flame, autoencoder, writer, save_path, dev):
    sum_loss = 0
    sum_recon_loss = 0
    for i, (mead_motion, mead_template, emotion_one_hot, id_one_hot, file_name) in enumerate(val_loader):
        mead_motion = mead_motion.to(dev)
        mead_template = mead_template.to(dev)
        emotion_one_hot = emotion_one_hot.to(dev)

        motion = torch2mesh(flame, mead_motion[:, :, :50], mead_motion[:, :, 50:])
        template = torch2mesh(flame, mead_template[:, :, :50], mead_template[:, :, 50:])

        recon, quant_loss, info = autoencoder(motion, template, emotion_one_hot)

        # LOSS
        loss, loss_details = vq_loss(recon, motion, quant_loss)

        sum_loss += loss.item()
        sum_recon_loss += loss_details[0].item()
    
    return sum_loss / len(val_loader), sum_recon_loss / len(val_loader)



def vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = torch.nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

def recone_loss(output_motion, motion):
    loss_l1 = torch.nn.functional.l1_loss(output_motion, motion)
    return loss_l1

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

# 冻结某一模型
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

if __name__ == "__main__":
    main()

# videos = torch.randn(1, 3, 5, 32, 32) # video (batch, channels, frames, height, width) - normalized from -1 to +1
# loss = diffusion(videos)
# loss.backward()
# # after a lot of training

# sampled_videos = diffusion.sample(batch_size = 4)
# sampled_videos.shape # (4, 3, 5, 32, 32)