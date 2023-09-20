import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data
from models.wav2vec import Wav2Vec2Model

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train",read_audio=False):
        self.data = data
        self.copy = 5 # 一个动作复制几次，用于增加数据量
        self.len = len(self.data) * self.copy  # 按照motion获得长度
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        self.max_audio_length = 160000
        self.max_motion_length = 300  # BIWI最长为245，在此处直接设置为300
        self.audio_embed = 2 # 获取前后的音频特征
        self.motion_embed = 2 # 获取之前的运动特征

    def __getitem__(self, index):

        index = index % self.copy # 一个动作复制几次，用于增加数据量

        if self.data_type == "train":
            audio = self.data[index]["audio"]  # [1, n*2, 768]
            vertice = self.data[index]["vertice"]  # [n, 70110]
            template = self.data[index]["template"]  # [70110]
            file_name = self.data[index]["name"]

            audio = torch.autograd.Variable(audio, requires_grad = False)
            audio = audio.data

            return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(vertice), file_name
        
        elif self.data_type == "valid" or self.data_type == "test":
            audio = self.data[0]
            vertice = self.data[1]
            template = self.data[2]

            audio = torch.autograd.Variable(audio, requires_grad = False)
            audio = audio.data
            
            return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template)
        
    def __len__(self):
        return self.len * self.copy
    
def rand_collate_fn(batch):
    list = [item[0] for item in batch]

    min_length = np.inf
    for i in list:
        if i.size(1) // 2 <= min_length:
            min_length = i.size(1) // 2

    rand = np.random.randint(0, min_length)  # [0, min_length)
    audio_list = torch.stack([item[0][:,:(rand + 1) * 2,:].squeeze(0) for item in batch])
    motion_list = torch.stack([item[1][rand,:] for item in batch])
    template_list = torch.stack([item[2] for item in batch])
    past_motion_list = torch.stack([item[3][:rand,:] for item in batch])
    filename_list = [item[4] for item in batch]


    return audio_list, motion_list, template_list, past_motion_list, filename_list


def read_data():
    data_root = '/data/WX/BIWI_dataset/'
    wav_path = 'wav/F1'
    text_path = 'raw_text'
    vertices_path = 'vertices_npy'
    template_file = 'templates.pkl'
    wav2vec2model_path = '/data/WX/wav2vec2-base-960h'
    train_subjects = 'F2 F3 F4 M3 M4 M5'
    val_subjects = 'F2 F3 F4 M3 M4 M5'
    test_subjects = 'F2 F3 F4 M3 M4 M5'
    read_audio = True

    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(data_root, wav_path)
    text_path = os.path.join(data_root, text_path)
    vertices_path = os.path.join(data_root, vertices_path)

    processor = Wav2Vec2Processor.from_pretrained(wav2vec2model_path)
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2model_path)
    audio_encoder.eval()

    template_file = os.path.join(data_root, template_file)

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                if read_audio:
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    audio_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
                key = f.replace("wav", "npy")
                text_values = open(os.path.join(text_path, f[:-4] + '.txt')).read() # 加载原始文本
                data[key]['text'] = text_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

                audio_feature = audio_encoder(audio_values, 'BIWI', frame_num=data[key]["vertice"].shape[0]).last_hidden_state
                data[key]["audio"] = audio_feature if read_audio else None

                audio_length = audio_feature.shape[1]
                motion_length = data[key]["vertice"].shape[0]
                if audio_length > 2 * motion_length:
                    audio_feature = audio_feature[:,:2 * motion_length,:]
                elif audio_length < 2 * motion_length:
                    data[key]["vertice"] = data[key]["vertice"][:int(audio_length / 2),:]

                    
    subjects_dict = {}
    subjects_dict["train"] = [i for i in train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in test_subjects.split(" ")]

    splits = {'train':range(1,40),'val':range(40,41),'test':range(40,41)}

    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(batch_size=8, workers=10):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data()
    train_data = Dataset(train_data, subjects_dict, "train", True)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=rand_collate_fn) # shuffle表示是否打乱顺序
    valid_data = Dataset(valid_data, subjects_dict, "valid", True)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=workers)
    test_data = Dataset(test_data, subjects_dict, "test", True)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
