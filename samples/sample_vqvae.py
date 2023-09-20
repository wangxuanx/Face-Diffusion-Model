import argparse
import sys

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm, trange
from datasets.data_loader_frames import get_dataloaders
from models.fdm_vq import FDM
from models.encoder_decoder import Motion_Encoder, Motion_Decoder
from video_diffusion_pytorch.video_diffusion_pytorch import Unet3D, GaussianDiffusion
from models.vq_vae import VQAutoEncoder

def main():

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vq_vae/biwi_stage1.pth.tar')['state_dict'])

    model = FDM(feature_dim=128, vertice_dim=70110, struct='Enc')

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )
    load_diffusion('./checkpoints/diffusion_vqvae', '2000', diffusion)

    save_path = './checkpoints/diffusion_vqvae/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估diffusion模型
    autoencoder.eval()  # 评估vqvae模型
    diffusion.to(dev)
    autoencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10)
    test_loader = loader['test']

    # 对数据进行采样
    sample_step(test_loader, dev, diffusion, autoencoder, 1, len(test_loader), save_path)


def sample_step(test_loader, dev, diffusion, autoencoder, epoch_log, epochs, save_folder):
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
         for n, (audio, motion, template) in enumerate(test_loader):
            n_frames = audio.shape[1] // 2  # 生成的帧总数
            template = template.to(dev).unsqueeze(1).flatten(-2)
            
            audio_ids = [0, 1] * 3
            # for i in range((2 + 1) * 2):
            #     audio_ids += [i]

            motion_frames = torch.cat([template for _ in range(2)], dim=1)

            # 逐帧采样生成
            result = []
            for j in trange(n_frames, desc=f'Sampling'):
                audio_cond = audio[:,audio_ids,:].to(dev)
                motion_frames = motion_frames.to(dev)

                enc_motion, _ = autoencoder.get_quant(torch.cat([motion_frames, template], dim=1) - template)
                sampled_frame = diffusion.sample(audio_cond, enc_motion[:, :, 16:], enc_motion[:, :, :16])
                sampled_frame = torch.cat([sampled_frame, sampled_frame, sampled_frame], dim=2)
                feat_out_q, _, _ = autoencoder.quantize(sampled_frame)
                # feature decoding
                output_motion = autoencoder.decode(feat_out_q)
                output_motion = output_motion[:, -1:, :] + template
                result.append(output_motion.cpu().detach().numpy())

                motion_frames = torch.cat([motion_frames[:, 1:, :], output_motion], dim=1)
                audio_ids = audio_ids[2:] + [min(j * 2 + 2, n_frames * 2 - 2)] + [min(j * 2 + 3, n_frames * 2 - 1)]


            result = np.concatenate(result, axis=0)

            np.save(os.path.join(save_folder, 'F2_e40_' + str(n) + '.npy'), result)

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

    print('freeze encoder decoder')
    freeze(encoder)
    freeze(decoder)

def load_diffusion(load_path, epoch, diffusion):
    print(f'load diffusion checkpoint from {load_path}/model-{epoch}.mpt')
    checkpoint = torch.load(str(load_path + f'/model-{epoch}.mpt'))['model']
    
    print('load diffusion')
    diffusion.load_state_dict(checkpoint, strict=False)
    print('load diffusion success')

# 冻结某一模型
def freeze(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

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


if __name__ == "__main__":
    main()