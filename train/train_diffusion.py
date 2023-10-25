import os
import sys
sys.path.append(".")
import torch
from tqdm import tqdm
from video_diffusion_pytorch.diffusion_encoder_decoder import Unet3D, GaussianDiffusion
from models.fdm import FDM
from datasets.data_loader import get_dataloaders
from torch.cuda.amp import autocast, GradScaler

from models.encoder_decoder import Motion_Encoder, Motion_Decoder

from torch.utils.tensorboard import SummaryWriter

def main():
    motion_enc = Motion_Encoder(1024, 70110)
    motion_dec = Motion_Decoder(1024, 70110)

    load_encoder_decoder('./checkpoints_code_error/encoder_decoder_load_model', '400', motion_enc, motion_dec)

    model = FDM(feature_dim=1024, vertice_dim=1024, struct='Dec')

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )

    load_model = False
    dev = 'cuda:1'
    diffusion.train()
    diffusion.to(dev)
    # 对Motion Encoder和Motion Decoder进行微调
    motion_enc.train()
    motion_dec.train()
    motion_enc.to(dev)
    motion_dec.to(dev)

    train_loader = get_dataloaders(batch_size=1, workers=10, read_audio=True, type="train")


    train_epoch = 2000
    optimizer = torch.optim.AdamW([{'params': diffusion.parameters(), 'lr': 0.0001},
                                   {'params': motion_dec.parameters(), 'lr': 0.0001},
                                   {'params': motion_enc.parameters(), 'lr': 0.0001}])

    save_path = './checkpoints/diffusion_Encoder_Decoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    if load_model:
        print('load model from checkpoints')
        load('./checkpoints/diffusion_frames', '100', diffusion, optimizer)

    for epoch in range(train_epoch):
        epoch_log = epoch + 1
        print(f'Starting epoch {epoch_log}')

        loss = run_step(train_epoch, epoch_log, optimizer, train_loader, diffusion, motion_enc, motion_dec, writer, save_path, dev)
        writer.add_scalar('Loss/train_epoch', loss, epoch_log)

def run_step(epochs, epoch_log, optimizer, train_loader, diffusion, encoder, decoder, writer, save_path, dev):
    
    with tqdm(range(len(train_loader)), desc=f'Train[{epoch_log}/{epochs}]') as tbar:
         for i, (audio, motion, template, one_hot, file_name) in enumerate(train_loader):
            optimizer.zero_grad()

            audio = audio.to(dev)
            motion = motion.to(dev)
            template = template.to(dev)
            one_hot = one_hot.to(dev)

            latent_motion = encoder(motion - template)
            denoise_loss, result = diffusion(latent_motion, audio, one_hot)

            output_motion = decoder(result) + template

            loss_recon = recone_loss(output_motion, motion[:,:output_motion.shape[1]])

            loss = denoise_loss * 1.0 + loss_recon * 1.0

            tbar.set_postfix(loss_recon=loss_recon.item(), denoise_loss=denoise_loss.item(), loss=loss.item())  # 为tqdm添加loss项
            tbar.update(1)

            loss.backward()
            optimizer.step()

            move_recone = torch.sum(output_motion)
            move_origin = torch.sum(motion)

            writer.add_scalar('Loss/denoise', denoise_loss, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/recon', loss_recon, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/train', loss, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/face_move', move_recone, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/face_move_origin', move_origin, (epoch_log - 1) * len(train_loader) + i)

    if epoch_log % 100 == 0:
        save(save_path, epoch_log, diffusion, encoder, decoder, optimizer)
    
    return loss

def recone_loss(output_motion, motion):
    # loss_l1 = torch.nn.functional.l1_loss(output_motion, motion)
    loss_l2 = torch.nn.functional.mse_loss(output_motion, motion)
    return loss_l2

def save(save_path, epoch, model, encoder, decoder, opt):
        data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
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