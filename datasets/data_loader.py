import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 
from torch.nn.utils.rnn import pad_sequence

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        self.copy = 1 # 一个动作复制几次，用于增加数据量

    def __getitem__(self, index):
        index = index % self.len # 一个动作复制几次，用于增加数据量
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        text = self.data[index]["text"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]

        subject = "_".join(file_name.split("_")[:-1])
        if self.data_type == "train":
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels[self.subjects_dict["val"].index(subject)]
        elif self.data_type == "test":
            one_hot = self.one_hot_labels[self.subjects_dict["test"].index(subject)]


        vertice = vertice.astype(np.float16)
        template = template.astype(np.float16)
        
        if self.read_audio:
            return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len * self.copy # 一个动作复制几次，用于增加数据量

def read_data(type="test"):
    data_root = '/data/WX/BIWI_dataset/'
    wav_path = 'wav'
    text_path = 'raw_text'
    vertices_path = 'vertices_npy'
    template_file = 'templates.pkl'
    wav2vec2model_path = '/data/WX/wav2vec2-base-960h'
    train_subjects = 'F2 F3 F4 M3 M4 M5'
    val_subjects = 'F2 F3 F4 M3 M4 M5'
    test_subjects = 'F1 F5 F6 F7 F8 M1 M2 M6'
    read_audio = True

    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    if type == "train":
        sub = [i for i in train_subjects.split(" ")]
    elif type == "val":
        sub = [i for i in val_subjects.split(" ")]
    elif type == "test":
        sub = [i for i in test_subjects.split(" ")]

    
    audio_path = os.path.join(data_root, wav_path)
    text_path = os.path.join(data_root, text_path)
    vertices_path = os.path.join(data_root, vertices_path)
    if read_audio: # read_audio==False when training vq to save time
        processor = Wav2Vec2Processor.from_pretrained(wav2vec2model_path)

    template_file = os.path.join(data_root, template_file)

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav") and check_in_list(sub, f):
                if read_audio:
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    processed = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
                    input_values = np.squeeze(processed)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values if read_audio else None
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

    subjects_dict = {}
    subjects_dict["train"] = [i for i in train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in test_subjects.split(" ")]


    #train vq and pred
    splits = {'train':range(1,37),'val':range(37,41),'test':range(37,41)}


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

def check_in_list(sub, f):
    for i in sub:
        if i in f:
            return True
    return False

def padding_collate_fn(batch):
    batch_audio_list = [item[0] for item in batch]
    batch_motion_list = [item[1] for item in batch]
    batch_template_list = [item[2] for item in batch]
    batch_onehot_list = [item[3] for item in batch]
    batch_filename_list = [item[4] for item in batch]
    padding_audio = pad_sequence(batch_audio_list, batch_first=True, padding_value=0)
    padding_motion = pad_sequence(batch_motion_list, batch_first=True, padding_value=0)
    padding_template = pad_sequence(batch_template_list, batch_first=True, padding_value=0)
    padding_onehot = pad_sequence(batch_onehot_list, batch_first=True, padding_value=0)

    result = []
    result.append(padding_audio)
    result.append(padding_motion)
    result.append(padding_template)
    result.append(padding_onehot)
    result.append(batch_filename_list)

    return result


def get_dataloaders(batch_size=64, workers=10, read_audio=False, type="train"):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(type=type)
    if type == "train":
        train_data = Dataset(train_data, subjects_dict, "train", read_audio)
        dataset = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    elif type == "val":
        valid_data = Dataset(valid_data,subjects_dict,"val",read_audio)
        dataset = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=workers)
    elif type == "test":
        test_data = Dataset(test_data,subjects_dict,"test",read_audio)
        dataset = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
