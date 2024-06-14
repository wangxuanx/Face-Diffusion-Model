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
        if data_type == "train":
            self.len = self.data[1].shape[0]  # 按照motion获得长度
        elif data_type == "val" or data_type == "test":
            self.len = 1
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        self.max_audio_length = 160000
        self.max_motion_length = 300  # BIWI最长为245，在此处直接设置为300
        self.copy = 1 # 一个动作复制几次，用于增加数据量

    def __getitem__(self, index):
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
        return self.len


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
                    data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)

                audio_feature = audio_encoder(audio_values, frame_num=data[key]["vertice"].shape[0]).last_hidden_state
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


    #train vq and pred
    # splits = {'vocasRet':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
    # 'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
    splits = {'train':range(1,39),'val':range(39,40),'test':range(39,40)}


    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(v)

    # for k, v in data.items():
    #     train_data.append(v)
    #     valid_data.append(v)
    #     test_data.append(v)
    train_audio = []
    train_vertice = []
    for i in train_data:
        train_audio.append(i["audio"].squeeze(0))
        train_vertice.append(i["vertice"])
    train_audio = torch.concatenate(train_audio, axis=0)
    train_vertice = np.concatenate(train_vertice, axis=0)
    n = train_vertice.shape[0]
    train_vertice = train_vertice.reshape((n, -1, 3))
    train = [train_audio, train_vertice, templates['F2']]
    

    val_audio = []
    val_vertice = []
    for i in valid_data:
        val_audio.append(i["audio"].squeeze(0))
        val_vertice.append(i["vertice"])
    val_audio = torch.concatenate(val_audio, axis=0)
    val_vertice = np.concatenate(val_vertice, axis=0)
    n = val_vertice.shape[0]
    val_vertice = val_vertice.reshape((n, -1, 3))
    val = [val_audio, val_vertice, templates['F2']]

    test_audio = []
    test_vertice = []
    for i in test_data:
        test_audio.append(i["audio"].squeeze(0))
        test_vertice.append(i["vertice"])
    test_audio = torch.concatenate(test_audio, axis=0)
    test_vertice = np.concatenate(test_vertice, axis=0)
    n = test_vertice.shape[0]
    test_vertice = test_vertice.reshape((n, -1, 3))
    test = [test_audio, test_vertice, templates['F2']]

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(train_vertice.shape[0], val_vertice.shape[0], test_vertice.shape[0]))
    return train, val, test, subjects_dict

def get_dataloaders(batch_size=64, workers=10):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data()
    train_data = Dataset(train_data, subjects_dict, "train", True)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_data = Dataset(valid_data, subjects_dict, "valid", True)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=workers)
    test_data = Dataset(test_data, subjects_dict, "test", True)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()