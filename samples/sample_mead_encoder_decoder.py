import sys
import librosa

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm, trange
from datasets.data_loader_mead import get_dataloaders
from models.fdm_ae_encoder_decoder import FDM
from models.encoder_decoder import Motion_Encoder, Motion_Decoder
from video_diffusion_pytorch.diffusion_mead_encoder_decoder import Unet3D, GaussianDiffusion
from models.wav2vec import Wav2Vec2Model

from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from utiles.flame_utils import get_mesh, torch2mesh
from transformers import Wav2Vec2Processor

import warnings
warnings.filterwarnings('ignore')

def main():
    flame_config = get_config()
    flame = FLAME(flame_config)  # 加载FLAME模型

    audioencoder = Wav2Vec2Model.from_pretrained('/data/WX/wav2vec2-base-960h')
    motion_enc = Motion_Encoder(512, 5023 * 3)
    motion_dec = Motion_Decoder(512, 5023 * 3)

    load_encoder_decoder('./checkpoints/mead_encoder_decoder_all_train', '2', motion_enc, motion_dec)

    model = FDM(feature_dim=512, vertice_dim=5023 * 3, struct='Dec')
    # model = ClassifierFreeSampleModel(model)

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    load_diffusion('./checkpoints/diffusion_mead_encoder_decoder', '600', diffusion)

    save_path = './checkpoints/diffusion_mead_encoder_decoder/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估模型
    flame.eval()
    motion_enc.eval()
    motion_dec.eval()
    audioencoder.eval()
    diffusion.to(dev)
    flame.to(dev)
    motion_enc.to(dev)
    motion_dec.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    test_loader = loader['test']

    # predict(audioencoder, diffusion, motion_dec, dev)

    # 对数据进行采样
    sample_step(test_loader, dev, diffusion, audioencoder, flame, motion_dec, 1, len(test_loader), save_path)

@torch.no_grad()
def sample_step(test_loader, dev, diffusion, audioencoder, flame, decoder, epoch_log, epochs, save_folder):
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
          for n, (audio, _, template, emo_one_hot, id_one_hot, file_name) in enumerate(test_loader):
            audio = audio.to(dev)
            template = template.to(dev)
            emo_one_hot = emo_one_hot.to(dev)
            id_one_hot = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            id_one_hot = id_one_hot.to(dev)

            template = torch2mesh(flame, template[:, :, :50], template[:, :, 50:])

            length = audioencoder(audio).last_hidden_state.shape[1] // 2
            # 逐帧采样生成
            result = diffusion.sample(audio, (1, length, 512), emo_one_hot, id_one_hot)

            output_motion = decoder(result) + template
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
    wav_path = '/data/WX/fdm/sample_out/music.wav'
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained('/data/WX/wav2vec2-base-960h')
    processed = processor(speech_array, sampling_rate=16000).input_values
    input_values = np.squeeze(processed)
    emotion_one_hot = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0])
    one_hot = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    audio = torch.from_numpy(input_values).unsqueeze(0)
    template = torch.from_numpy(np.load('/data/WX/MEAD/FLAME_template.npy'))
    audio = audio.to(dev)
    template = template.to(dev).flatten(-2)
    emotion_one_hot = emotion_one_hot.to(dev)
    one_hot = one_hot.to(dev)

    shape = audioencoder(audio).last_hidden_state.shape
    # 逐帧采样生成
    result = diffusion.sample(audio, (1, shape[1] // 2, 512), emotion_one_hot, one_hot)

    output_motion = decoder(result) + template
    output_motion = output_motion.detach().cpu().numpy()

    np.save(os.path.join('/data/WX/fdm/sample_out', 'music'), output_motion)


if __name__ == "__main__":
    main()