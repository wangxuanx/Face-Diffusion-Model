import sys

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm, trange
from datasets.data_loader_frames import get_dataloaders
from models.fdm import FDM
from models.encoder_decoder import Motion_Encoder, Motion_Decoder
from video_diffusion_pytorch.video_diffusion_pytorch import Unet3D, GaussianDiffusion


def main():

    motion_enc = Motion_Encoder(1024, 70110)
    motion_dec = Motion_Decoder(1024, 70110)

    load_encoder_decoder('./checkpoints/encoder_decoder_load_model', 'best', motion_enc, motion_dec)

    model = FDM(feature_dim=1024, vertice_dim=1024, struct='Enc')

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )
    load_diffusion('./checkpoints/diffusion', '2000', diffusion)

    save_path = './checkpoints/diffusion/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估模型
    diffusion.to(dev)
    motion_enc.to(dev)
    motion_dec.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10)
    test_loader = loader['test']

    # 对数据进行采样
    sample_step(test_loader, dev, diffusion, motion_enc, motion_dec, 1, len(test_loader), save_path)


def sample_step(test_loader, dev, diffusion, encoder, decoder, epoch_log, epochs, save_folder):
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
         for n, (audio, motion, template) in enumerate(test_loader):
            n_frames = audio.shape[1] // 2  # 生成的帧总数
            template = template.to(dev).unsqueeze(1).flatten(-2)
            
            audio_ids = [0, 1] * 2
            # for i in range((2 + 1) * 2):
            #     audio_ids += [i]

            motion_frames = torch.cat([template for _ in range(1)], dim=1)

            # 逐帧采样生成
            result = []
            for j in trange(n_frames, desc=f'Sampling'):
                audio_cond = audio[:,audio_ids,:].to(dev)
                motion_frames = motion_frames.to(dev)

                latent_motion = encoder(template - template)
                latent_motion_frames = encoder(motion_frames - template)

                sampled_frame = diffusion.sample(audio_cond, latent_motion, latent_motion_frames)

                output_motion = decoder(sampled_frame) + template
                result.append(output_motion.cpu().numpy())

                motion_frames = torch.cat([motion_frames, output_motion], dim=1)
                audio_ids = audio_ids[2:] + [min((j + 2) * 2 + 2, n_frames * 2 - 2)] + [min((j + 2) * 2 + 3, n_frames * 2 - 1)]
                # audio_ids = audio_ids[:] + [(j + 2) * 2 + 2]

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


if __name__ == "__main__":
    main()