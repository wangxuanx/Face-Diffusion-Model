import sys
import librosa

import numpy as np
sys.path.append(".")
import os
import torch
from tqdm import tqdm
from datasets.data_loader_vocaset import get_dataloaders
from models.fdm_vocaset import FDM
from video_diffusion_pytorch.diffusion_BIWI_encoder_decoder import GaussianDiffusion
from models.hubert import HubertModel

from transformers import Wav2Vec2Processor

from models.utils.config import vocaset_vq_vae_args
from models.vq_vae_vocaset import VQAutoEncoder

import warnings
warnings.filterwarnings('ignore')

def main():

    vq_args = vocaset_vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)
    autoencoder.load_state_dict(torch.load('checkpoints/vqvae_vocaset/model.pth.tar')['state_dict'])
    audioencoder = HubertModel.from_pretrained('hubert-large-ls960-ft')
    model = FDM(feature_dim=1024)
    # model = ClassifierFreeSampleModel(model)

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )

    load_diffusion('./checkpoints/diffusion_VOCASET_vqvae', '100', diffusion)

    save_path = './checkpoints/diffusion_VOCASET_vqvae/result'
    dev = 'cuda:1'
    diffusion.eval()  # 评估模型
    autoencoder.eval()
    audioencoder.eval()
    diffusion.to(dev)
    autoencoder.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 创建保存路径
        os.makedirs(save_path)

    loader = get_dataloaders(batch_size=1, workers=10, read_audio=True)
    val_loader = loader['test']

    # predict(audioencoder, diffusion, motion_dec, dev)

    # 对数据进行采样
    sample_step(val_loader, dev, diffusion, audioencoder, autoencoder, 1, len(val_loader), save_path)

@torch.no_grad()
def sample_step(test_loader, dev, diffusion, audioencoder, autoencoder, epoch_log, epochs, save_folder):
    train_subject = 'FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA'
    subjects = train_subject.split(' ')

    sr = 16000
    with tqdm(range(len(test_loader)), desc=f'Sampling [{epoch_log}/{epochs}]') as tbar:
        for n, (audio, _, template, one_hot_all, file_name) in enumerate(test_loader):
            audio = audio.to(dev)
            template = template.to(dev)
            one_hot_all = one_hot_all.to(dev)

            for iter in range(one_hot_all.shape[-1]):
                condition_subject = subjects[iter]
                id_one_hot = one_hot_all[:,iter,:]
                
                # num_frames = int(audio.shape[-1] / sr * 30)
                num_frames = audioencoder(audio, 'vocaset').last_hidden_state.shape[1]
                # 逐帧采样生成

                # result = diffusion.sample(audio, (1, length * 8, 128), id_one_hot)
                result = diffusion.ddim_sample(audio, (1, num_frames * 16, 64), id_one_hot, 100)

                quanted, _, _ = autoencoder.quant(result)
                output_motion = autoencoder.decode(quanted) + template
                output_motion = output_motion.detach().cpu().numpy()

                id = torch.argmax(id_one_hot)
                style = subjects[id]
                np.save(os.path.join(save_folder, file_name[0][:-4] + '_condition_' + style), output_motion)


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