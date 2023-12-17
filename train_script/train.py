
import argparse
import os
import pickle

#from azureml.core import Workspace
from azureml.core.run import Run
#from azureml.core import Dataset

import json
from math import ceil
import torch
import torchaudio
from torch.utils.data import DataLoader
# output will be logged, separate output from previous log entries.
print('-'*100)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
                        dest='data_path', 
                        default='data', 
                        help='data folder mounting point')

    return parser.parse_args()

def make_chunks_torch(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(audio_segment.shape[1] / float(chunk_length))
    for i in range(int(number_of_chunks)):
        yield audio_segment[0,i * chunk_length:(i + 1) * chunk_length]

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,input_data,transform=None):
        self.input_data = input_data
        self.transform=transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        label = self.input_data[index][0]
        audiochunk = self.input_data[index][1]
        mfcc = self.input_data[index][2]
        if self.transform:
            audiochunk = self.transform(audiochunk)
        return audiochunk,mfcc,label


if __name__ == '__main__':

    # parse the parameters passed to the this script
    args = parse_args()

    # set data paths
    data_folder = args.data_path
    print("Data Location:",data_folder)

    print("Loading data...")

    run = Run.get_context()
    #ws = run.experiment.workspace

    #
    CHANNELS = 1
    RATE =  32000
    CHUNK = 1024
    #

    audio_data = []
    folder_labels = dict()
    num_classes=0

    folder = 'Audio' #Assume this is where all audio folders are located. This folder is at same level as script.
    abspath = folder
    subfolders = os.listdir(abspath)

    trans = torchaudio.transforms.MFCC(RATE)
    for audiogroup in subfolders:
        print("Folder:",audiogroup)
        files = os.listdir(os.path.join(abspath,audiogroup))
        folder_labels[audiogroup] = num_classes
        #print("\tFiles:",files)
        for file in files:
            relpath = os.path.join(os.path.join(abspath,audiogroup),file)
            #print(f"\t\tLoading {relpath}")
            data,samplerate = torchaudio.load(relpath,normalize=False)
            if samplerate!=RATE:
                data = torchaudio.transforms.Resample(samplerate,RATE)(data)
            data = torch.mean(data,dim=0).unsqueeze(0)
            data = data / torch.max(data)
            for chunk in make_chunks_torch(data,CHUNK):
                torchdatachunk = torch.cat([chunk,torch.zeros(CHUNK-len(chunk))])
                if torch.numel(torchdatachunk[torch.abs(torchdatachunk)>.001]) >= CHUNK//2: #remove silence
                    mfcc = trans(torchdatachunk)
                    torchdatachunk = torch.reshape(torchdatachunk,(1,-1))
                    audio_data.append((num_classes,torchdatachunk,mfcc))
        num_classes+=1

    batch_size = 64
    train_ds = AudioDataset(audio_data[0:1])
    train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

    main_info = dict()
    main_info['folders'] = folder_labels
    main_info['channels'] = CHANNELS
    main_info['rate'] = RATE
    main_info['chunk'] = CHUNK
    with open("outputs/main_info.json", "w") as f:
        json.dump(main_info, f)
    print("Training...")

    '''# Create ImageGenerators
    print('Creating train ImageDataGenerator')
    train_generator = ImageDataGenerator(rescale=1/255)\
                            .flow_from_directory(train_folder, 
                                                 batch_size = 32)
    val_generator = ImageDataGenerator(rescale=1/255)\
                            .flow_from_directory(val_folder, 
                                                 batch_size = 32)

    # Build the model
    model = K.models.Sequential()
    model.add(K.layers.Conv2D(32, (2,2), activation='relu'))
    model.add(K.layers.MaxPooling2D(2,2))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # fit model and store history
    history = model.fit(train_generator, 
                        validation_data=val_generator,
                        epochs=10)

    print('Saving model history...')
    with open(f'outputs/model.history', 'wb') as f:
        pickle.dump(history.history, f)

    print('Saving model history...')
    model.save(f'outputs/model.model')'''

    print('Done!')
    print('-'*100)