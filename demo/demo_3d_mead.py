import argparse
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
    flame = FLAME(flame_config)  # load the FLAME model

    vq_args = vq_vae_args()
    autoencoder = VQAutoEncoder(vq_args)

    model_args = get_args()
    autoencoder.load_state_dict(torch.load(model_args.stage1_model_path)['model'])

    audioencoder = Wav2Vec2Model.from_pretrained('wav2vec2-base-960h')
    model = FDM(feature_dim=model_args.feature_dim, vertice_dim=model_args.vertice_dim, struct='Dec')
    # model = ClassifierFreeSampleModel(model)

    diffusion = GaussianDiffusion(
        model,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )

    load_diffusion(model_args.stage2_model_path, diffusion)  # load the diffusion model

    save_path = model_args.result_path
    dev = model_args.device

    diffusion.eval()  # 评估模型
    flame.eval()
    autoencoder.eval()
    audioencoder.eval()
    diffusion.to(dev)
    flame.to(dev)
    autoencoder.to(dev)
    audioencoder.to(dev)

    if not os.path.exists(save_path): # 
        os.makedirs(save_path)


    predict(diffusion, autoencoder, dev, model_args.audio_file, model_args)


def load_diffusion(load_path, diffusion):
    print(f'load diffusion checkpoint from {load_path}')

    checkpoint = torch.load(load_path)['model']
    
    print('load diffusion')
    diffusion.load_state_dict(checkpoint, strict=False)
    print('load diffusion success')
    

@torch.no_grad()
def predict(diffusion, autoencoder, dev, audio_path, args):
    sr = 16000
    audio_name = audio_path.split('/')[-1]

    speech_array, sampling_rate = librosa.load(audio_path, sr=sr)
    processor = Wav2Vec2Processor.from_pretrained('/data/WX/wav2vec2-base-960h')
    processed = processor(speech_array, sampling_rate=16000).input_values
    input_values = np.squeeze(processed)
    emotion_one_hot = torch.FloatTensor([0, 0, 0, 0, 1, 0, 0])  # ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    one_hot = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0
                                ])
    
    input_values = np.concatenate((input_values, np.zeros(16000 * 1).astype(np.float32)))
    audio = torch.from_numpy(input_values).unsqueeze(0)
    template = torch.from_numpy(np.load('/data/WX/MEAD/FLAME_template.npy'))
    audio = audio.to(dev)
    template = template.to(dev).flatten(-2)
    emotion_one_hot = emotion_one_hot.to(dev)
    one_hot = one_hot.to(dev)

    num_frames = int(audio.shape[-1] / sr * 24)

    result = diffusion.sample(audio, (1, num_frames * 8, 64), emotion_one_hot, one_hot)

    quanted, _, _ = autoencoder.quant(result, emotion_one_hot)
    output_motion = autoencoder.decode(quanted) + template
    output_motion = output_motion.detach().cpu().numpy()

    np.save(os.path.join(args.audio_path, audio_name[:-4]), output_motion)
        

def get_args():
    parser = argparse.ArgumentParser(description='Expressive 3D Facial Animation Generation Based on Local-to-global Latent Diffusion')
    parser.add_argument("--audio_file", type=str, help='the audio file path for prediction')
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices')
    parser.add_argument("--feature_dim", type=int, default=512, help='the fdm feature dimension')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--stage1_model_path", type=str, default="3D_MEAD/3d_mead_stage1.mpt", help='path of the stage 1 pretrained models')
    parser.add_argument("--stage2_model_path", type=str, default="3D_MEAD/3d_mead_stage2.mpt", help='path of the stage 2 pretrained models')
    parser.add_argument("--audio_path", type=str, default="3D_MEAD/result", help='path to the predictions')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()