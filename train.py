from speechsplit import Generator
import matplotlib.pyplot as plt
import torch
import torchaudio
import pyaudio
import torch.nn.functional as F
import wave
import numpy as np
from pydub import AudioSegment
from math import ceil
import os
import sounddevice as sd
import time
import datetime
from testa import Data
from pathlib import Path
import librosa

class HParams():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    

FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1
CHUNK = 1024
OUTPUT_DEVICE = sd.default.device[1]
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
            aud =audio_segment[i * chunk_length:(i + 1) * chunk_length]
            if len(aud) < chunk_length:
                yield audio_segment[i * chunk_length:]
            else:
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


def train():
    g_lr = 0.001
    beta1,beta2 = .9,.999
    num_iters = 1000

    
    hparams = HParams(
        # model   
        freq = 8,
        dim_neck = 8,
        freq_2 = 8,
        dim_neck_2 = 1,
        freq_3 = 8,
        dim_neck_3 = 32,
        out_channels = 10 * 3,
        layers = 24,
        stacks = 4,
        residual_channels = 512,
        gate_channels = 512,  # split into 2 groups internally for gated activation
        skip_out_channels = 256,
        cin_channels = 80,
        gin_channels = -1,  # i.e., speaker embedding dim
        weight_normalization = True,
        n_speakers = -1,
        dropout = 1 - 0.95,
        kernel_size = 3,
        upsample_conditional_features = True,
        upsample_scales = [4, 4, 4, 4],
        freq_axis_kernel_size = 3,
        legacy = True,
        
        dim_enc = 512,
        dim_enc_2 = 128,
        dim_enc_3 = 256,
        
        dim_freq = 80,
        dim_spk_emb = 82,
        dim_f0 = 257,
        dim_dec = 512,
        len_raw = 128,
        chs_grp = 16,
        
        # interp
        min_len_seg = 19,
        max_len_seg = 32,
        min_len_seq = 64,
        max_len_seq = 128,
        max_len_pad = 192,
        
        # data loader
        root_dir = 'assets/spmel',
        feat_dir = 'assets/raptf0',
        batch_size = 16,
        mode = 'train',
        shuffle = True,
        num_workers = 0,
        samplier = 8,
        
        # Convenient model builder
        builder = "wavenet",

        hop_size = 256,
        log_scale_min = float(-32.23619130191664),
    )

    G = Generator(hparams)

    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    G.to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1,beta2])

    # Start training.
    start_iters = 1
    print('Start training...')
    start_time = time.time()
    for i_iter in range(start_iters, num_iters+1):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        # Fetch real images and labels.
        audio_list = []

        #source_path = Path("Audio/Kindred_1")
        #for audiofile in os.listdir(source_path):
        #    audio_list.append(source_path / audiofile)
        audio_list.append(Path("Audio/Kindred_1/Kindred.attack01.wav"))
        #rand_num = torch.distributions.uniform.Uniform(.5,1.5)
        for file in audio_list:
            print("Training on:",file)
            dat = Data(file)
            S_org,mel_org,pitch_org = dat.audio,dat.spec,dat.pitch
            spec_min = dat.spec_min
            spec_max = dat.spec_max
            assert spec_max != None and spec_min != None

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            G = G.train()

            # Identity mapping loss
            #x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            #x_f0_intrp = torch.nn.functional.interpolate(x_f0,scale_factor=rand_num.sample().item()) 
       
            label = 0

            orig_sound = mel_org.to(device).float()

            orig_resample = mel_org.to(device).float()
            orig_pitch = pitch_org.to(device).unsqueeze(0).float()

            pred_sound = G(orig_sound,orig_resample,orig_pitch,label)
            #assert orig_sound.shape == pred_sound.shape

            g_loss_id = F.mse_loss(pred_sound,orig_sound,reduction='mean') 
            
            # Backward and optimize.
            g_loss = g_loss_id
            g_optimizer.zero_grad()
            #print(pred_sound.shape,orig_sound.shape)
            g_loss.backward()
            g_optimizer.step()
        

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            print("g_loss:",g_loss.item())

            if i_iter % 4 == 0:    
                fig, axs = plt.subplots(4,1)
                print(pred_sound.shape)
                rec_mel = pred_sound[0,:,:].detach().numpy()
                im = axs[0].imshow(rec_mel,origin="lower", aspect="auto")
                im2 = axs[1].imshow(orig_sound[0,:,:].detach().numpy(), origin="lower", aspect="auto")
                fig.colorbar(im, ax=axs)


                def denormalize(data,newmin=-1,newmax=1):
                    min = torch.min(data)
                    max = torch.max(data)
                    data = (data - min) / (max - min)
                    newdata = data * (newmax - newmin) + newmin
                    assert np.isclose(torch.min(newdata).item(),newmin) and np.isclose(torch.max(newdata).item(),newmax)
                    return newdata
                deform = denormalize(pred_sound,spec_min,spec_max)

                inverse_melscale_transform = torchaudio.transforms.InverseMelScale(n_stft=1024 // 2 + 1,n_mels=80,sample_rate=16000)
                spectrogram = inverse_melscale_transform(deform.detach())
                transform = torchaudio.transforms.GriffinLim(n_fft=1024)
                waveform = transform(spectrogram)

                #mod =  torchaudio.functional.DB_to_amplitude(deform,ref=1,power=1)
                #im = axs[2].plot(S_org[0,:].detach().numpy())
                #rec_audio = librosa.feature.inverse.mel_to_audio(mod.detach().numpy(),sr=16000,n_fft=1024,)
                #im2 = axs[3].plot(rec_audio[0,:])
                #plt.show()

                print(waveform.shape,S_org.shape)
                #assert waveform.shape == S_org.shape
                
                dat.play_audio(S_org.detach().numpy())

                filename = "test2.wav"
                paudio = pyaudio.PyAudio()
                waveFile = wave.open(filename, 'wb')
                waveFile.setnchannels(1)
                waveFile.setsampwidth(paudio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)

                #chunks = []
                #for i in range(0,S_org.shape[1], CHUNK):
                #    chunks.append(S_org[i:i+CHUNK])
                #waveFile.writeframes(b''.join(np.asarray(chunks*5000).astype(np.int16).flatten()))
                #waveFile.close()
                break


train()