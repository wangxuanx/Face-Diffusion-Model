import argparse
import os
import sys

sys.path.append(".")
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from video_diffusion_pytorch.video_diffusion_pytorch import Unet3D, GaussianDiffusion
from models.fdm_vq import FDM
from datasets.data_loader import get_dataloaders
from torch.cuda.amp import autocast, GradScaler

from models.vq_vae import VQAutoEncoder
from models.wav2vec import Wav2Vec2Model

from torch.utils.tensorboard import SummaryWriter

def vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=128, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=70110, help='number of input channels')
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
    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vq_vae/biwi_stage1.pth.tar')['state_dict'])
    for param in autoencoder.parameters():
        param.requires_grad = False

    # 加载vaw2vec2.0
    audioencoder = Wav2Vec2Model.from_pretrained('/data/WX/wav2vec2-base-960h')
    # wav2vec 2.0 weights initialization
    audioencoder.feature_extractor._freeze_parameters()

    model = FDM(feature_dim=128, vertice_dim=70110, struct='Enc')

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )

    load_model = False
    dev = 'cuda:0'

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    train_loader = loader['train']

    train_epoch = 2000
    optimizer = torch.optim.AdamW([{'params':diffusion.parameters(), 'lr':0.00001},
                                   {'params':audioencoder.parameters(), 'lr':0.00001}])

    save_path = './checkpoints/diffusion_vqvae_squence_4_prior'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    if load_model:
        print('load model from checkpoints')
        load('./checkpoints/diffusion_signal_frame', '100', diffusion, optimizer)

    diffusion.train()
    audioencoder.train()
    diffusion.to(dev)
    audioencoder.to(dev)
    autoencoder.to(dev)

    for epoch in range(train_epoch):
        epoch_log = epoch + 1
        print(f'Starting epoch {epoch_log}')

        loss = run_step(train_epoch, epoch_log, optimizer, train_loader, diffusion, audioencoder, autoencoder, writer, save_path, dev)
        print(f'Epoch {epoch_log} loss: {loss}')
        writer.add_scalar('Loss/train_epoch', loss, epoch_log)

def run_step(epochs, epoch_log, optimizer, train_loader, diffusion, audioencoder, autoencoder, writer, save_path, dev):
    batch_size = 32
    sum_loss = 0
    with tqdm(range(len(train_loader)), desc=f'Train[{epoch_log}/{epochs}]') as tbar:
         for i, (audio, motion, template, one_hot, file_name) in enumerate(train_loader):
            optimizer.zero_grad()

            num_frames = motion.shape[1]
            audio = audioencoder(audio.to(dev), frame_num=num_frames).last_hidden_state
            # 从一个序列中挑选出batchsize大小的数据进行训练
            audio, motion = get_audio_motion(audio, motion.to(dev), batch_size)  

            template = template.to(dev).unsqueeze(1).flatten(-2)
            one_hot = one_hot.to(dev)
            # [BATCH, 128, 8*num_frames]
            enc_motion, _ = autoencoder.get_quant(motion - template)  # [batch, 128, 8]

            denoise_loss, result = diffusion(enc_motion[:, :, -8:], audio, enc_motion[:, :, :-8], one_hot)
            
            # vq_loss = vq_vae_loss(result, enc_motion)
            # feature quantization
            feat_out_q, _, _ = autoencoder.quantize(result)
            # feature decoding
            output_motion = autoencoder.decode(torch.cat([feat_out_q, feat_out_q, feat_out_q], dim=1))
            output_motion = torch.mean(output_motion, dim=1) + template

            loss_recon = recone_loss(output_motion, motion[:,-1,:])

            # tbar.set_postfix(vq_loss=vq_loss.item(), noise_loss=denoise_loss.item(), loss_recon=loss_recon.item(), loss=loss.item())  # 为tqdm添加loss项
            tbar.set_postfix(loss_recon=loss_recon.item(), noise_loss=denoise_loss.item())
            tbar.update(1)

            sum_loss += denoise_loss.item()

            loss = loss_recon + denoise_loss 
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/denoise', denoise_loss, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/vq_loss', vq_loss, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/recon', loss_recon, (epoch_log - 1) * len(train_loader) + i)
            writer.add_scalar('Loss/train', loss, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/face_move', move_recone, (epoch_log - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss/face_move_origin', move_origin, (epoch_log - 1) * len(train_loader) + i)

    if epoch_log % 100 == 0:
        save(save_path, epoch_log, diffusion, audioencoder, optimizer)
    
    return sum_loss / len(train_loader)

# 从一个数据中随机选择batchsize大小的序列
def get_audio_motion(all_audio, all_motion, batchsize):
    # 将audio和motion的长度统一
    if all_audio.shape[1] // 2 > all_motion.shape[1]:
        all_audio = all_audio[:, :all_motion.shape[1]]
    elif all_audio.shape[1] // 2 < all_motion.shape[1]:
        all_motion = all_motion[:, :all_audio.shape[1] // 2]
    num_frames = all_audio.shape[1] // 2
    assert num_frames == all_motion.shape[1] # 保证audio和motion的长度一致

    # 随机选择batchsize大小的序列
    random_idx = torch.randint(num_frames, (batchsize,))
    idx = random_idx.unsqueeze(1)
    audio_idx = torch.cat([(idx - 2) * 2, (idx - 2) * 2 + 1, (idx - 1) * 2, (idx - 1) * 2 + 1, idx * 2, idx * 2 + 1, (idx + 1) * 2, (idx + 1) * 2 + 1, (idx + 2) * 2, (idx + 2) * 2 + 1], dim=1)
    motion_idx = torch.cat([(idx - 4), (idx - 3), (idx - 2), (idx - 1), idx], dim=1)

    audio_idx[audio_idx < 0] = 0
    audio_idx[audio_idx > num_frames * 2 - 1] = num_frames * 2 - 1
    motion_idx[motion_idx < 0] = 0
    motion_idx[motion_idx > num_frames - 1] = num_frames - 1

    audio = all_audio[:, audio_idx, :]  # 五帧音频用来训练
    motion = all_motion[:, motion_idx, :]  # 取额外的四帧来训练

    audio = audio.squeeze(0)
    motion = motion.squeeze(0)
    return audio, motion

def vq_vae_loss(output, target):
    loss = torch.nn.functional.mse_loss(output, target)
    return loss

def recone_loss(output_motion, motion):
    loss_l2 = torch.nn.functional.mse_loss(output_motion, motion)
    return loss_l2

def save(save_path, epoch, model, audioencoder, opt):
        data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'audioencoder': audioencoder.state_dict(),
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