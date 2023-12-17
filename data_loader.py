from pathlib import Path
from utils import *
import math
from scipy.io import wavfile
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import torchaudio
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyaudio
from torchvision import transforms

import matplotlib.pyplot as plt
import custommodels as cm
import seaborn as sns

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        if type(index)==int:
            data_info = self.input_data[index]
            label = np.asarray([data_info[0]]).astype(np.float32)
            audiochunk = read_np(data_info[1]).astype(np.float32)
        else:
            print("WARNING: slicing the dataloader can lead to large memory allocation.")
            data_info = self.input_data[index]
            label = np.asarray([d[0] for d in data_info]).astype(np.float32)
            audiochunk = np.asarray([read_np(d[1]) for d in data_info]).astype(np.float32)
        return audiochunk,label
    

def make_chunks_torch(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = math.ceil(audio_segment.shape[1] / float(chunk_length))
    for i in range(int(number_of_chunks)):
        yield audio_segment[0,i * chunk_length:(i + 1) * chunk_length]




OUTPUT_DEVICE = sd.default.device[1]
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE =  32000
CHUNK = 1024



def load_audio_file(path):
    data,samplerate = torchaudio.load(path,normalize=False)
    data = data.to(torch.float32)
    if samplerate!=RATE:
        data = torchaudio.transforms.Resample(samplerate,RATE)(data)
    data = torch.mean(data,dim=0).unsqueeze(0)
    biggestval = torch.max(torch.abs(data))
    data = data / biggestval
    data = torch.reshape(data,(1,-1))
    return data



class AudioDat(torch.utils.data.Dataset):
    def __init__(self,input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        data_info = self.input_data[index]
        f1 = data_info[0]
        f2 = data_info[1]
        return f1,f2
def load():
    irange = range(0,15) #15
    f1 = 0
    f2 = 1

    datlist = []
    for i in irange:
        file1 = f"F/M/{f1}_{i}.wav"
        file2 = f"F/F/{f2}_{i}.wav"

        audio1 = load_audio_file(file1)
        audio2 = load_audio_file(file2)
        setlength = max(audio1.shape[1],audio2.shape[1])
        audio1 = torch.cat([audio1,torch.zeros(1,setlength-audio1.shape[1])],axis=1).type(torch.float32)#.unsqueeze(-2)
        audio2 = torch.cat([audio2,torch.zeros(1,setlength-audio2.shape[1])],axis=1).type(torch.float32)#.unsqueeze(-2)
        datlist.append((audio1,audio2))

    loader = AudioDat(datlist)
    t_loader = DataLoader(loader, batch_size=1,shuffle=True)

    return t_loader



def load_data(class_names=None):
    loader = AudioDataset
    source_folder = Path("Audio")
    save_folder = Path("data_folder")
    save_file = "audio_data.gz"
    subfolder_name = "data"


    audio_data = {}
    if class_names==None:
        _, dirs, _ = next(os.walk(source_folder))
        class_names = dirs

    for class_name in class_names:
        print(f"Loading class {class_name}...")
        class_num = int(class_name.rsplit("_",1)[1])
        if not (source_folder / class_name).exists():
            raise FileNotFoundError(f"Folder {class_name} not found in {source_folder}")
        if all_files_exist([save_folder / class_name / save_file]):
            print(f"Loading data from {save_file}")
            all_data = read_json(save_folder / class_name / save_file) 
            audio_data[class_num] = all_data[f"{class_num}"]
        else:
            print(f" {save_file} not found. Creating new data.")
            ensure_folder_exists(save_folder / class_name)
            audio_files = os.listdir(source_folder / class_name)
            new_audio_file_paths = {}
            new_audio_file_paths[class_num] = []

            for file in audio_files:
                relpath = source_folder / class_name / file
                data,samplerate = torchaudio.load(relpath,normalize=False)
                data = data.to(torch.float32)
                if samplerate!=RATE:
                    data = torchaudio.transforms.Resample(samplerate,RATE)(data)
                data = torch.mean(data,dim=0).unsqueeze(0)
                data = data / torch.max(data)
                #data = (data - torch.mean(data)) / torch.std(data) 
                #data = torch.nn.functional.normalize(data)
                chunknum = 0
                datalength = data.shape[1] #CHUNK
                for chunk in make_chunks_torch(data,datalength): #CHUNK
                    torchdatachunk = torch.cat([chunk,torch.zeros(datalength-len(chunk))])
                    #if torch.numel(torchdatachunk[torch.abs(torchdatachunk)>.001]) >= datalength//2: #remove silence
                    torchdatachunk = torch.reshape(torchdatachunk,(1,-1))
                    save_np(torchdatachunk,save_folder / class_name / subfolder_name, f"{file}_{chunknum}")
                    new_audio_file_paths[class_num].append(f"{save_folder / class_name / subfolder_name / file}_{chunknum}.gz")
                    chunknum+=1
            save_json(new_audio_file_paths,save_folder / class_name,save_file)
            audio_data[class_num]=new_audio_file_paths[class_num]
    finaldata = []
    for k,v in audio_data.items():
        for path in v:
            finaldata.append((k,path))

    return finaldata,loader