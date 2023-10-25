import sys
import librosa

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm, trange
from datasets.data_loader import get_dataloaders
from models.fdm import FDM
from models.encoder_decoder import Motion_Encoder, Motion_Decoder
from video_diffusion_pytorch.diffusion_encoder_decoder import Unet3D, GaussianDiffusion
from models.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Processor


def main():
    audioencoder = Wav2Vec2Model.from_pretrained('/data/WX/wav2vec2-base-960h')
    motion_enc = Motion_Encoder(1024, 70110)
    motion_dec = Motion_Decoder(1024, 70110)

    load_encoder_decoder('./checkpoints/diffusion_Encoder_Decoder', '1000', motion_enc, motion_dec)

    model = FDM(feature_dim=1024, vertice_dim=1024, struct='Dec')

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        num_frames = 5,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    load_diffusion('./checkpoints/diffusion_Encoder_Decoder', '1000', diffusion)

    save_path = './checkpoints/diffusion_Encoder_Decoder/new_result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估模型
    motion_enc.eval()
    motion_dec.eval()
    audioencoder.eval()
    diffusion.to(dev)
    motion_enc.to(dev)
    motion_dec.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    test_loader = get_dataloaders(batch_size=1, workers=10, read_audio=True, type="val")

    predict(audioencoder, diffusion, motion_dec, dev)

    # 对数据进行采样
    # sample_step(test_loader, dev, diffusion, audioencoder, motion_enc, motion_dec, 1, len(test_loader), save_path)

@torch.no_grad()
def sample_step(test_loader, dev, diffusion, audioencoder, encoder, decoder, epoch_log, epochs, save_folder):
    wav_path = '/data/WX/fdm/sample_out/music.wav'
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained('/data/WX/wav2vec2-base-960h')
    processed = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
    # input_values = np.squeeze(processed)

    audio = torch.FloatTensor(processed)

    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
         for n, (_, motion, template, one_hot, file_name) in enumerate(test_loader):
            audio = audio.to(dev)
            template = template.to(dev).unsqueeze(1).flatten(-2)
            one_hot = one_hot.to(dev)

            shape = audioencoder(audio).last_hidden_state.shape
            # 逐帧采样生成
            result = diffusion.sample(audio, (1, shape[1] // 2, 1024), one_hot)

            output_motion = decoder(result) + template
            output_motion = output_motion.detach().cpu().numpy()

            np.save(os.path.join(save_folder, file_name[0][:-4]), output_motion)

# 加载Motion Encoder和Motion Decoder
def load_encoder_decoder(load_path, epoch, encoder, decoder):
    print(f'load encoder decoder checkpoint from {load_path}/model-{epoch}.mpt')
    encoder_dict = torch.load(str(load_path + f'/model-{epoch}.mpt'))['encoder']
    decoder_dict = torch.load(str(load_path + f'/model-{epoch}.mpt'))['decoder']
        
    encoder.load_state_dict(encoder_dict, strict=False)
    decoder.load_state_dict(decoder_dict, strict=False)

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

def predict(audioencoder, diffusion, decoder, dev):
    wav_path = '/data/WX/fdm/sample_out/music.mp3'
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained('/data/WX/wav2vec2-base-960h')
    processed = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
    input_values = np.squeeze(processed)

    audio = torch.from_numpy(input_values)
    template = torch.load('/data/WX/fdm/sample_out/template.pt')
    audio = audio.to(dev)
    template = template.to(dev).unsqueeze(1).flatten(-2)
    one_hot = one_hot.to(dev)

    shape = audioencoder(audio).last_hidden_state.shape
    # 逐帧采样生成
    result = diffusion.sample(audio, (1, shape[1] // 2, 1024), one_hot)

    output_motion = decoder(result) + template
    output_motion = output_motion.detach().cpu().numpy()

    np.save(os.path.join('/data/WX/fdm/sample_out', 'music'), output_motion)


if __name__ == "__main__":
    main()