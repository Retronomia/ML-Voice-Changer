import torchaudio
import torchvision
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pysptk as sptk
from numpy.random import RandomState
import pyaudio
import wave
from pydub import AudioSegment
import sounddevice as sd
from math import ceil

FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
OUTPUT_DEVICE = sd.default.device[1]
CHANNELS = 1
RATE =  16000

import os
from scipy import signal
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Data:
    def __init__(self,file_path,sample_rate=16000,chunk=1024,duration=5):

        self.duration=duration*sample_rate
        self.sample_rate = sample_rate
        self.chunk = chunk

        self.audio = self.load(file_path)
        self.spec_max = None
        self.spec_min = None
        self.spec = self.make_spectrogram()
        self.pitch = self.make_pitch()
    
    def load(self,file_path):
        file_p = Path(file_path)
        data,samplerate = torchaudio.load(file_p,normalize=False)
        data = data.to(torch.float32)

        data = data[:,:self.duration]
        if data.shape[1] < self.duration:
            data = torch.cat([data,torch.zeros(1,self.duration-data.shape[1])],dim=1)

        if samplerate!=RATE:
            data = torchaudio.transforms.Resample(samplerate,self.sample_rate)(data)

        if data.shape[0] > 1:
            data = torch.mean(data, dim=0, keepdim=True)
        return data
    
    def normalize(self,data,newmin=-1,newmax=1):
        min = torch.min(data)
        max = torch.max(data)
        data = (data - min) / (max - min)
        newdata = data * (newmax - newmin) + newmin
        assert np.isclose(torch.min(newdata),newmin) and np.isclose(torch.max(newdata),newmax)
        return newdata
    
    def make_spectrogram(self):
        spec_pipe = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.chunk, n_mels=80)
        spec = spec_pipe(self.audio).to(torch.float32)
        amp = torchaudio.transforms.AmplitudeToDB('power')
        spec = amp(spec)
        #mel_data = 20 * np.log10(np.maximum(1e-6, mel_data)) - 16
        self.spec_min = torch.min(spec).item()
        self.spec_max = torch.max(spec).item()
        spec = self.normalize(spec)
        return spec
        
    def make_pitch(self):
        f0_rapt = torchaudio.functional.detect_pitch_frequency(self.audio,sample_rate=self.sample_rate)
        return f0_rapt
    
    def play_audio(self,audio=None):
        if audio is None:
            audio = self.audio.detach().numpy()
        flat_audio = (audio[0,:]*10000).astype(np.int16).flatten()
        p = pyaudio.PyAudio()
        outputstream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output_device_index=OUTPUT_DEVICE,
                        output=True,start=False)

        outputstream.start_stream()

        seg = AudioSegment(
            flat_audio.tobytes(), 
            frame_rate=RATE,
            sample_width=pyaudio.get_sample_size(FORMAT), 
            channels=CHANNELS
        )
        outputstream.write(seg._data)
        outputstream.stop_stream()


        filename = "test2.wav"
        paudio = pyaudio.PyAudio()
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(paudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)

        waveFile.writeframesraw(flat_audio.tobytes())
        waveFile.close()

    def plot(self):
        fig,ax = plt.subplots(3,1)
    
        end_time = self.audio.shape[1] / self.sample_rate
        time_axis = torch.linspace(0, end_time, self.audio.shape[1])
        ax[0].plot(time_axis, self.audio[0], linewidth=1, color="gray", alpha=0.3)

        time_axis = torch.linspace(0, end_time, self.pitch.shape[1])
        ax[1].plot(time_axis, self.pitch[0], linewidth=2, label="Pitch", color="green")
        ax[1].legend(loc=0)

        #librosa.power_to_db(
        im = ax[2].imshow(self.spec[0,:], origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax[2])
        plt.show()



def playAudio(audioflat):
    def make_chunks(audio_segment, chunk_length):
        """
        Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
        long.
        if chunk_length is 50 then you'll get a list of 50 millisecond long audio
        segments back (except the last one, which can be shorter)
        """
        number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
        for i in range(int(number_of_chunks)):
            #aud =audio_segment[i * chunk_length:(i + 1) * chunk_length]
            #if len(aud) < chunk_length:
            #    yield audio_segment[i * chunk_length:]
            #else:
            yield audio_segment[i * chunk_length:(i + 1) * chunk_length]
    p = pyaudio.PyAudio()
    outputstream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output_device_index=OUTPUT_DEVICE,
                    output=True,start=False)

    outputstream.start_stream()

    seg = AudioSegment(
        audioflat.tobytes(), 
        frame_rate=RATE,
        sample_width=pyaudio.get_sample_size(FORMAT), 
        channels=CHANNELS
    )
    for chunk in make_chunks(seg,CHUNK):
        outputstream.write(chunk._data)

    outputstream.stop_stream()


if  __name__ == '__main__':
    d = Data("Audio/Kindred_1/Kindred.attack01.wav")
    d.play_audio()
    #d.plot()
    #playAudio((d.audio[0,:]*10000).detach().numpy().astype(np.int16).flatten())


# def load_audio_file(path):
#     data = Data(path)



#     return raw_audio,mel_data,pitch_data


# def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or "Spectrogram (db)")
#     axs.set_ylabel(ylabel)
#     axs.set_xlabel("frame")
#     im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
#     fig.colorbar(im, ax=axs)
#     plt.show()

# print("Start")
# filed = load_audio_file(Path("Audio/Kindred_1/Kindred.attack01.wav"))
# print("Loaded Data.")
# results = filed
# sound = results[0][0,:1000]
# result2 = results[1]
# result3 = results[2]
# #filtered = results[1][0,:1000]
# #plt.plot(sound,color='red',alpha=0.5)
# print(result3.shape)
# plt.plot(result3[0,:])
# #plot_spectrogram(result2[0,:,:])
# #plt.plot(result2[0,0,:],result2[0,1,:])
# #plt.plot(filtered,color='blue',alpha=0.5)
# plt.show()
# #plt.savefig('/tmp/test.png')
# #plt.close()
# print("End")


