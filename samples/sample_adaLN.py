import argparse
import sys

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm, trange
from datasets.data_loader import get_dataloaders
from models.fdm_adaLN import FDM
from models.encoder_decoder import Motion_Encoder, Motion_Decoder
from video_diffusion_pytorch.diffusion_adaLN import Unet3D, GaussianDiffusion
from models.vq_vae import VQAutoEncoder
from models.wav2vec import Wav2Vec2Model

def main():

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vq_vae/biwi_stage1.pth.tar')['state_dict'])
    # 加载vaw2vec2.0
    audioencoder = Wav2Vec2Model.from_pretrained('/data/WX/wav2vec2-base-960h')

    model = FDM(feature_dim=1024) 

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    load_diffusion('./checkpoints/diffusion_vqvae_adaLN', '200', diffusion)
    load_audioencoder('./checkpoints/diffusion_vqvae_adaLN', '200', audioencoder)

    save_path = './checkpoints/diffusion_vqvae_adaLN/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估diffusion模型
    autoencoder.eval()  # 评估vqvae模型
    audioencoder.eval()  # 评估audioencoder模型
    diffusion.to(dev)
    autoencoder.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=1, read_audio=True)
    test_loader = loader['test']

    # 对数据进行采样
    sample_step(test_loader, dev, diffusion, autoencoder, audioencoder, 1, len(test_loader), save_path)


def sample_step(test_loader, dev, diffusion, autoencoder, audioencoder, epoch_log, epochs, save_folder):
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
         for n, (audio, motion, template, one_hot, file_name) in enumerate(test_loader):
            
            template = template.to(dev).unsqueeze(1)

            audio = audioencoder(audio.to(dev)).last_hidden_state
            n_frames = audio.shape[1] // 2  # 生成的帧总数
            one_hot = one_hot.to(dev)

            # 逐帧采样生成
            result = []
            for j in trange(n_frames, desc=f'Sampling'):
                audio_cond = audio[:,[j * 2, j * 2 + 1],:].to(dev)

                # 使用template作为初始值来生成序列
                enc_motion, _ = autoencoder.get_quant(torch.cat([template, template, template], dim=1) - template)
                sampled_frame = diffusion.sample(audio_cond, enc_motion[:, :, -8:], one_hot)
                sampled_frame = torch.cat([sampled_frame, sampled_frame, sampled_frame], dim=2)
                feat_out_q, _, _ = autoencoder.quantize(sampled_frame)
                # feature decoding
                output_motion = autoencoder.decode(feat_out_q)
                output_motion = output_motion[:, -1:, :] + template
                result.append(output_motion.cpu().detach().numpy())


            result = np.concatenate(result, axis=0)

            np.save(os.path.join(save_folder, file_name[0][:-4]), result)

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

# 加载audioencoder
def load_audioencoder(load_path, epoch, audioencoder):
    print(f'load audioencoder checkpoint from {load_path}/model-{epoch}.mpt')
    checkpoint = torch.load(str(load_path + f'/model-{epoch}.mpt'))['audioencoder']
    
    print('load audioencoder')
    audioencoder.load_state_dict(checkpoint, strict=False)
    print('load audioencoder success')

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