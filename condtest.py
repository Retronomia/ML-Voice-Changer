from data_loader import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np
import wave


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

res,loader = load_data(['Ekko_0','Kindred_1']) #,'Kindred_1','Scout_9','Steve_10'])

batch_size = 1
train_ds = loader(res)

train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)



class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel,self).__init__()
        self.CHUNK = 1024
        self.hidden_len = self.CHUNK
        self.lstm = nn.LSTM(self.CHUNK+1,self.hidden_len, 1,batch_first=True)
    def forward(self,input,labels):
        N,C,T = input.shape
        num_iters = T // self.CHUNK
        hidden = torch.zeros(N,1,self.hidden_len).to(input.device)
        cell = torch.zeros(N,1,self.hidden_len).to(input.device)
        output = torch.tensor([], dtype=torch.float32, device=input.device)
        for i in range(num_iters):
            input_chunk = torch.cat([input[:,:,i*self.CHUNK:(i+1)*self.CHUNK],torch.unsqueeze(labels,dim=0)],dim=-1)
            #input_chunk = nn.ConstantPad1d((0, self.CHUNK - input_chunk.shape[2]), 0)(input_chunk)
            #print(input_chunk.shape)
            out,(hidden,cell) = self.lstm(input_chunk,(hidden,cell))
            output = torch.cat([output,out], dim=2)
        return output
    
    
    
model = CustomModel().to(device)
st_d = torch.load('model.pth')
model.load_state_dict(st_d)

model.eval()


'''with torch.no_grad():
    test_num=0
    for audio,labels in train_loader:
        audio = audio.to(device)
        tlabels = labels.to(device)
        
        reconaudio = []
        recon = model(audio,tlabels)
        reconaudio += list(recon.detach().cpu().numpy())
        reconaudio = (np.array(reconaudio)*5000).astype(np.int16).flatten()


        filename = f"samples\{int(labels[-1].item())}_{test_num}.wav"
        test_num+=1
        paudio = pyaudio.PyAudio()
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(paudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(reconaudio))
        waveFile.close()'''


with torch.no_grad():
    test_num=0
    for audio,labels in train_loader:
        audio = audio.to(device)
        tlabels = torch.tensor([[0]]).to(device) if int(labels[-1].item())==1 else torch.tensor([[1]]).to(device)
        
        reconaudio = []
        recon = model(audio,tlabels)
        reconaudio += list(recon.detach().cpu().numpy())
        reconaudio = (np.array(reconaudio)*5000).astype(np.int16).flatten()


        filename = f"condsamples\{int(labels[-1].item())}_{test_num}.wav"
        test_num+=1
        paudio = pyaudio.PyAudio()
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(paudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(reconaudio))
        waveFile.close()