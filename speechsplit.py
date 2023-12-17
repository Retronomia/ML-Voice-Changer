import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
class ConvNorm(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=None,dilation=1,bias=True,w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        #torch.nn.init.xavier_uniform_(
        #    self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



class Encoder_1(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutions for code 1
        convolutions = []
        for i in range(3):
            conv_layer = ConvNorm(1 if i==0 else 3,3,
                    kernel_size=5, stride=1,
                    padding=2,
                    dilation=1, w_init_gain='relu')
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)
        self.upsample = torch.nn.Upsample(scale_factor=.9)
        
        self.lstm_1 = nn.LSTM(3,8,2, batch_first=True, bidirectional=True)
        
    def forward(self,original):
        x = original
        for conv_layer in self.convolutions_1:
            x = F.relu(conv_layer(x))
        x = self.upsample(x)
        return x

class Encoder_2(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutions for code 1
        self.convolutions_1 = nn.Sequential(
            ConvNorm(80,32,
                kernel_size=5, stride=1,
                padding=2,
                dilation=1, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(32,16,
                kernel_size=5, stride=1,
                padding=2,
                dilation=1, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(16,3,
                kernel_size=5, stride=1,
                padding=2,
                dilation=1, w_init_gain='relu'),
            torch.nn.ReLU(),
            )

        #self.lstm_1 = nn.LSTM(3,8,2, batch_first=True, bidirectional=True)
        
    def forward(self,original):
        x = original
        for conv_layer in self.convolutions_1:
            x = conv_layer(x)
        return x


class Decoder_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions= nn.Sequential(
            ConvNorm(3,16),
            torch.nn.ReLU(),
            ConvNorm(16,32),
            torch.nn.ReLU(),
            ConvNorm(32,64),
            torch.nn.ReLU(),
            ConvNorm(64,80),
            torch.nn.Tanh(),
        )
        self.lstm_1 = nn.LSTM(80,9,2, batch_first=True, bidirectional=False)
        
    def forward(self,encoder_stack,orig_shape):
        x = encoder_stack
        for conv_layer in self.convolutions:
            x = conv_layer(x)
        #x,(h_n,c_n) = self.lstm_1(x)
        return x


class Generator(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        
        self.encoder_1 = Encoder_2()
        self.encoder_2 = Encoder_2()
        self.encoder_3 = Encoder_1()

        self.decoder = Decoder_1()


    def forward(self, original,resample,pitch,label):
        #print(original.shape,resample.shape,pitch.shape)

        encoder_outputs_1 = self.encoder_1(original)
        #encoder_outputs_2 = self.encoder_2(resample)
        #encoder_outputs_3 = self.encoder_3(pitch)



        encoder_stack = encoder_outputs_1
        #label_stack = torch.ones(encoder_outputs_1.shape) * label
        #print(encoder_outputs_1.shape,encoder_outputs_2.shape,encoder_outputs_3.shape,label_stack.shape)
        #encoder_stack = torch.cat((encoder_outputs_1,label_stack),dim=1)


        mel_output  = self.decoder(encoder_stack,original.shape)

        #print(mel_output.shape)

        return mel_output
    