import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.discriminator = nn.ModuleList([
            #conv layer
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1,16,kernel_size=15,stride=1)),
                nn.LeakyReLU(0.2,inplace=True)
                ),
            #x4 downsample
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16,64,kernel_size=41,stride=4,padding=20,groups=4)),
                nn.LeakyReLU(0.2,inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64,256,kernel_size=41,stride=4,padding=20,groups=4)),
                nn.LeakyReLU(0.2,inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256,1024,kernel_size=41,stride=4,padding=20,groups=4)),
                nn.LeakyReLU(0.2,inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024,1024,kernel_size=41,stride=4,padding=20,groups=4)),
                nn.LeakyReLU(0.2,inplace=True),
            ),
            #convs
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024,1024,kernel_size=5,stride=1,padding=2)),
                nn.LeakyReLU(0.2,inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024,1,kernel_size=3,stride=1,padding=1)),
        ])
    def forward(self,x):
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1],features[-1]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )
    def forward(self,x):
        ret = list()
        
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret # [(feat, score), (feat, score), (feat, score)]

class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class Generator(nn.Module):
    def __init__(self,n_channels):
        super(Generator, self).__init__()
        self.mel_channel = n_channels #mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(self.mel_channel, 512, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),

            ResStack(256),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),

            ResStack(64),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        MAX_WAV_VALUE = 32768.0
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return 