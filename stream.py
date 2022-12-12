import numpy as np
import pyaudio
import librosa
import time
from multiprocessing import Process
from multiprocessing import set_start_method
import os
from queue import Queue
from threading import Thread 
import struct
import wave
import threading
import matplotlib.pyplot as plt
import sounddevice as sd
from pydub import AudioSegment
from math import ceil
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024


#print(sd.query_devices(kind='input'))
#print(sd.query_devices())
#14 Headset Earphone (Arctis 5 Chat), Windows DirectSound (0 in, 2 out)


paudio = pyaudio.PyAudio()
print("Starting to record...")
stream = paudio.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=False,start=True,frames_per_buffer=CHUNK)
outputstream = paudio.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output_device_index=5,
                output=True,start=False) #p.get_format_from_width(seg.sample_width)

pll = []
start = time.time()
lenn = 0
while (time.time()-start) <= 2.5:
    data = stream.read(CHUNK)
    data_int =  np.frombuffer(data, dtype=np.int16)  #+ 127
    #print(data_int)
    #do stuff to data
    mod_data = data_int
    #do stuff to data
    lenn+= len(data_int)
    byte_data = mod_data.tobytes()
    pll.append(byte_data)


stream.stop_stream()
stream.close()
outputstream.start_stream()


def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length:(i + 1) * chunk_length]
            for i in range(int(number_of_chunks))]

seg = AudioSegment(
    b''.join(pll), 
    frame_rate=RATE,
    sample_width=pyaudio.get_sample_size(FORMAT), 
    channels=CHANNELS
)

# Just in case there were any exceptions/interrupts, we release the resource
# So as not to raise OSError: Device Unavailable should play() be used again

# break audio into half-second chunks (to allows keyboard interrupts)
for chunk in make_chunks(seg, 500):
    outputstream.write(chunk._data)

outputstream.stop_stream()
paudio.terminate()


print('-----Finished Recording-----')

'''# Open and Set the data of the WAV file
filename = "F:\AudioAI\soundsample.wav"
waveFile = wave.open(filename, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(paudio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(pll))
waveFile.close()
'''


#old
'''
def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length:(i + 1) * chunk_length]
            for i in range(int(number_of_chunks))]


seg = AudioSegment(
    b''.join(pll), 
    frame_rate=RATE,
    sample_width=pyaudio.get_sample_size(FORMAT), 
    channels=CHANNELS
)
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(seg.sample_width),
                channels=seg.channels,
                rate=seg.frame_rate,
                output_device_index=5,
                output=True)

# Just in case there were any exceptions/interrupts, we release the resource
# So as not to raise OSError: Device Unavailable should play() be used again
try:
    # break audio into half-second chunks (to allows keyboard interrupts)
    for chunk in make_chunks(seg, 500):
        stream.write(chunk._data)
finally:
    stream.stop_stream()
    stream.close()

    p.terminate()
'''


#playsound wav
'''# import required module
from playsound import playsound
f_name = 'F:\AudioAI\soundsample.wav'
  
# for playing note.wav file
playsound(f_name)
'''
#pydub wav
'''# import required modules
from pydub import AudioSegment
from pydub.playback import play
  
# for playing wav file
f_name = 'F:\AudioAI\soundsample.wav'
song = AudioSegment.from_wav(f_name)
print('playing sound using  pydub')
play(song)'''

#pydub bytes
'''from scipy.io.wavfile import read
from pydub import AudioSegment
import sounddevice as sd
sd.default.device = 14

audio_segment = AudioSegment(
    b''.join(pll), 
    frame_rate=RATE,
    sample_width=pyaudio.get_sample_size(FORMAT), 
    channels=CHANNELS
)

# test that it sounds right (requires ffplay, or pyaudio):
from pydub.playback import play
play(audio_segment)'''


#THREADING
'''
# Set up some global variables
num_fetch_threads = 2
enclosure_queue = Queue()

# A real app wouldn't use hard-coded data...
feed_urls = [ 'http://www.castsampler.com/cast/feed/rss/guest','asdsada','adsadsadasd','asdasdasdasd'
             ]


def downloadEnclosures(i, q):
    """This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    while True:
        print ('%s: Looking for the next enclosure' % i)
        url = q.get()
        print ('%s: Downloading:' % i, url)
        # instead of really downloading the URL,
        # we just pretend and sleep
        time.sleep(i + 2)
        q.task_done()


# Set up some threads to fetch the enclosures
for i in range(num_fetch_threads):
    worker = Thread(target=downloadEnclosures, args=(i, enclosure_queue,))
    worker.setDaemon(True)
    worker.start()

# Download the feed(s) and put the enclosure URLs into
# the queue.
for url in feed_urls:
    enclosure_queue.put(url)

#enclosure_queue.join()
print ('*** Done')'''