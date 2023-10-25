import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from models.wav2vec import Wav2Vec2Model
import librosa
from torch.nn.utils.rnn import pad_sequence

import pandas as pd


def reparse_emotion_name(emotion):
    emotion_dict = {
        'angry': ['angry', 'anger'],
        'contempt': ['contempt'],
        'disgusted': ['disgusted', 'disgust'],
        'fear': ['fear', 'fearful'],
        'happy': ['happy', 'happiness'],
        'sad': ['sad', 'sadness'],
        'surprised': ['surprised', 'surprise'],
        'neutral': ['neutral']
    }
    for k, v in emotion_dict.items():
        if emotion in v:
            return k
    else:
        raise ValueError('emotion name {} not found'.format(emotion))


def label_to_idx(label_type: str, label: str, one_hot: bool = False):
    if label_type == 'emotion':
        label = reparse_emotion_name(label)
        if label == 'neutral':
            if one_hot:
                return torch.zeros(7)
            else:
                return -1
        emotion_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        assert label in emotion_list, 'emotion label should be in {}, found {}'.format(emotion_list, label)
        if one_hot:
            return torch.eye(len(emotion_list))[emotion_list.index(label)]
        else:
            return emotion_list.index(label)
    elif label_type == 'speaker':
        speaker_list = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023',
                        'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033',
                        'M034', 'M035', 'M037', 'M039', 'M040']
        assert label in speaker_list, 'speaker label should be in {}, found {}'.format(speaker_list, label)
    
        if one_hot:
            return torch.eye(len(speaker_list))[speaker_list.index(label)]
        else:
            return speaker_list.index(label)
    
    else:
        raise ValueError('label type {} not found'.format(label_type))


class MEADDataset(data.Dataset):
    def __init__(self, df, is_train: bool, p_general: float = 0.3, read_audio: bool = False):
        super().__init__()
        self.dataset_root = '/data/WX/MEAD'
        self.subjects_dict = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        self.is_train = is_train
        self.p_general = p_general
        self.read_audio = read_audio
        self.df = df
        self.one_hot_labels = np.eye(len(self.subjects_dict))

    def __getitem__(self, index):
        row = self.df.iloc[index]  # [pid, emotion, intensity, flame_id, audio_id]
        if row['intensity'] == 'level_3':
            audio_path = os.path.join(self.dataset_root, 'AUDIO', row['pid'], row['emotion'],
                                    row['intensity'], row['audio_id'])
            flame_path = os.path.join(self.dataset_root, 'FLAME_ALL', row['pid'], 
                                    f"{row['pid']}-{row['emotion']}-{row['intensity']}-{row['flame_id']}")
            if self.read_audio:
                processor = Wav2Vec2Processor.from_pretrained('/data/WX/wav2vec2-base-960h')
                audio, _ = librosa.load(audio_path, sr=16000)
                processed = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
                audio_values = np.squeeze(processed)

            file_name = row['pid'] + '_' + row['emotion'] + '_' + row['intensity'] + '_' + row['audio_id']
            flame = self.get_flame(flame_path)
            
            if self.is_train:
                emotion_label = label_to_idx('emotion', row['emotion'], one_hot=True)
            else:
                # emotion_label = self.one_hot_labels
                emotion_label = label_to_idx('emotion', row['emotion'], one_hot=True)

            if self.is_train:
                speaker_label = label_to_idx('speaker', row['pid'], one_hot=True)
            else:
                speaker_label = label_to_idx('speaker', row['pid'], one_hot=True)

            template = torch.zeros((1, 56))  # 没有动作作为template

            if self.read_audio:
                return audio_values, flame, template, emotion_label, speaker_label, file_name
            else:
                return flame, template, emotion_label, speaker_label, file_name

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_audio(audio_path: str):
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio)
        length = torch.tensor(audio.shape[0])
        return audio, length

    @staticmethod
    def get_flame(flame_path: str):
        data = np.load(flame_path, allow_pickle=True)
        expression = torch.from_numpy(data['expression'])  # (T, 50)
        pose = torch.from_numpy(data['pose'])[:, 3:]  # (T, 3)
        pose = torch.cat([torch.zeros_like(pose), pose], dim=1)  # (T, 6)])
        return torch.cat([expression, pose], dim=1)  # (T, 56)


def padding_collate_fn(batch):
    batch_audio_list = [item[0] for item in batch]
    batch_motion_list = [item[1] for item in batch]
    batch_template_list = [item[2] for item in batch]
    batch_onehot_list = [item[3] for item in batch]
    # batch_filename_list = [item[4] for item in batch]
    padding_audio = pad_sequence(batch_audio_list, batch_first=True, padding_value=0)
    padding_motion = pad_sequence(batch_motion_list, batch_first=True, padding_value=0)
    padding_template = pad_sequence(batch_template_list, batch_first=True, padding_value=0)
    padding_onehot = pad_sequence(batch_onehot_list, batch_first=True, padding_value=0)

    result = []
    result.append(padding_audio)
    result.append(padding_motion)
    result.append(padding_template)
    result.append(padding_onehot)
    # result.append(batch_filename_list)

    return result

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(batch_size=64, workers=10, read_audio=False, type="train"):
    df = pd.read_csv('/data/WX/MEAD/mead_v2.csv')
    df = df[(df['audio_id'] == '001.m4a') | (df['audio_id'] == '002.m4a')]  # 此句单独使用001与002进行测试
    # need ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019']
    # df = df[df['emotion'] == 'angry']
    # df = df[(df['pid'] == 'M003') | (df['pid'] == 'M005') | (df['pid'] == 'M007') | (df['pid'] == 'M009') | (df['pid'] == 'M011') | (df['pid'] == 'M012') | (df['pid'] == 'M013') | (df['pid'] == 'M019') | (df['pid'] == 'M022') | (df['pid'] == 'M023')]
    # df = df[df['emotion'] == 'neutral']
    
    val_df = df[(df['pid'] == 'M039') | (df['pid'] == 'M035')]  # 验证集
    test_df = df[(df['pid'] == 'M040') | (df['pid'] == 'M037')]  # 测试集

    # pids = ['M003', 'M005', 'M007', 'M009', 'M011', 'W009', 'W011', 'W014', 'W015', 'W016']
    # train_df = pd.concat([df[df['pid'] == pid] for pid in pids])

    train_df = df.drop(val_df.index).drop(test_df.index)  # 将测试集和验证集从训练集中去除

    dataset = {
        'train': data.DataLoader(MEADDataset(train_df, is_train=True, read_audio=read_audio),
                                 batch_size=batch_size, shuffle=True,
                                 drop_last=True, num_workers=workers),
        'valid': data.DataLoader(MEADDataset(val_df, is_train=False, read_audio=read_audio),
                                 batch_size=1, shuffle=False,
                                 drop_last=True),
        'test': data.DataLoader(MEADDataset(test_df, is_train=False, read_audio=read_audio),
                                batch_size=1, shuffle=False)
    }
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    
