from scipy.io import wavfile
import wave
import numpy as np
import torchaudio
import os
import pathlib
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyaudio
from math import ceil

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE =  32000
CHUNK = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_data = []
i=0

def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    for i in range(int(number_of_chunks)):
        yield audio_segment[i * chunk_length:(i + 1) * chunk_length]

sampleratecheck = None
folder = 'Audio' #Assume this is where all audio folders are located. This folder is at same level as script.
abspath = os.path.join(pathlib.Path(__file__).parent.resolve(),folder)
subfolders = os.listdir(abspath)
for audiogroup in subfolders:
    print("Folder:",audiogroup)
    files = os.listdir(os.path.join(abspath,audiogroup))
    print("\tFiles:",files)
    for file in files:
        relpath = os.path.join(os.path.join(abspath,audiogroup),file)
        print(f"\t\tLoading {relpath}")
        samplerate, data = wavfile.read(relpath)
        if sampleratecheck==None:
            sampleratecheck = samplerate
        else:
            if samplerate != sampleratecheck:
                raise ValueError(f"Sample rate {samplerate} differs from {sampleratecheck}.")
        for chunk in make_chunks(data,CHUNK):
            audio_data.append((i,np.hstack((chunk,[0]*(CHUNK-len(chunk)))).reshape(1,-1)))
        i+=1

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index][0],self.input_data[index][1]




batch_size = 200
train_ds = AudioDataset(audio_data)
train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)


print(audio_data)



my_x = np.array([[1.0,2],[3,4]]) # a list of numpy arrays
my_y = np.array([[4.],[2.]]) # another list of numpy arrays (targets)

tensor_x = torch.Tensor(my_x) # transform to torch tensor
tensor_y = torch.Tensor(my_y)

#x1_dataset = torch.tensor(my_x, dtype=torch.float)

#x2_dataset = torch.tensor(my_y, dtype=torch.float)

#x_dataset = torch.stack([x1_dataset, x2_dataset])

#print(x_dataset)