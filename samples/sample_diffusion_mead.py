import sys
import librosa

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm
from datasets.data_loader_mead import get_dataloaders
from models.fdm_vqvae_mead import FDM
from video_diffusion_pytorch.diffusion_mead_encoder_decoder import Unet3D, GaussianDiffusion
from models.wav2vec import Wav2Vec2Model

from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import get_mesh, torch2mesh
from transformers import Wav2Vec2Processor

from utiles.args import vq_vae_args
from models.vq_vae_emotion import VQAutoEncoder

import warnings
warnings.filterwarnings('ignore')

def main():
    flame_config = get_config()
    flame = FLAME(flame_config)  # 加载FLAME模型

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('./checkpoints/vqvae_mead/model-30.mpt')['model'])

    audioencoder = Wav2Vec2Model.from_pretrained('wav2vec2-base-960h')
    model = FDM(feature_dim=512, vertice_dim=5023 * 3, struct='Dec')
    # model = ClassifierFreeSampleModel(model)

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )

    load_diffusion('./checkpoints/diffusion_mead_vqvae', '50', diffusion)

    save_path = './checkpoints/diffusion_mead_vqvae/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估模型
    flame.eval()
    autoencoder.eval()
    audioencoder.eval()
    diffusion.to(dev)
    flame.to(dev)
    autoencoder.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    val_loader = loader['test']

    # predict(audioencoder, diffusion, motion_dec, dev)

    # 对数据进行采样
    sample_step(val_loader, dev, diffusion, audioencoder, flame, autoencoder, 1, len(val_loader), save_path)

@torch.no_grad()
def sample_step(test_loader, dev, diffusion, audioencoder, flame, autoencoder, epoch_log, epochs, save_folder):
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
        for n, (audio, _, template, emo_one_hot, id_one_hot, file_name) in enumerate(test_loader):
            audio = audio.to(dev)
            template = template.to(dev)
            emo_one_hot = emo_one_hot.to(dev)
            id_one_hot = id_one_hot.to(dev)

            template = torch2mesh(flame, template[:, :, :50], template[:, :, 50:])

            length = audioencoder(audio).last_hidden_state.shape[1] // 2
            # 逐帧采样生成
            result = diffusion.sample(audio, (1, length * 8, 64), emo_one_hot, id_one_hot)

            quanted, _, _ = autoencoder.quant(result, emo_one_hot)
            output_motion = autoencoder.decode(quanted) + template
            output_motion = output_motion.detach().cpu().numpy()

            np.save(os.path.join(save_folder, file_name[0][:-4]), output_motion)

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



def load_diffusion(load_path, epoch, diffusion):
    print(f'load diffusion checkpoint from {load_path}/model-{epoch}.mpt')

    checkpoint = torch.load(str(load_path + f'/model-{epoch}.mpt'))['model']
    
    print('load diffusion')
    diffusion.load_state_dict(checkpoint, strict=False)
    print('load diffusion success')

def load_audioencoder(load_path, epoch, audioencoder):
    print(f'load audioencoder checkpoint from {load_path}/model-{epoch}.mpt')

    checkpoint = torch.load(str(load_path + f'/model-{epoch}.mpt'))['audioencoder']
    
    print('load audioencoder')
    audioencoder.load_state_dict(checkpoint, strict=False)
    print('load audioencoder success')

# 冻结某一模型
def freeze(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

@torch.no_grad()
def predict(audioencoder, diffusion, decoder, dev):
    wav_path = 'sample_out/music.wav'
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained('wav2vec2-base-960h')
    processed = processor(speech_array, sampling_rate=16000).input_values
    input_values = np.squeeze(processed)
    emotion_one_hot = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0])
    one_hot = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    audio = torch.from_numpy(input_values).unsqueeze(0)
    template = torch.from_numpy(np.load('MEAD/FLAME_template.npy'))
    audio = audio.to(dev)
    template = template.to(dev).flatten(-2)
    emotion_one_hot = emotion_one_hot.to(dev)
    one_hot = one_hot.to(dev)

    shape = audioencoder(audio).last_hidden_state.shape
    # 逐帧采样生成
    result = diffusion.sample(audio, (1, shape[1] // 2, 512), emotion_one_hot, one_hot)

    output_motion = decoder(result) + template
    output_motion = output_motion.detach().cpu().numpy()

    np.save(os.path.join('sample_out', 'music'), output_motion)


if __name__ == "__main__":
    main()