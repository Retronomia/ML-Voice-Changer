import pyaudio
import numpy as np
import threading
from pydub import AudioSegment
from math import ceil
import keyboard
import queue
import time
import sounddevice as sd

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE =  32000
CHUNK = 1024

INPUT_DEVICE = sd.default.device[0]
OUTPUT_DEVICE = sd.default.device[1] #4
EXIT_KEY = 'q'

#print(sd.query_devices(kind='output'))

print("Recording...")
def stream_audio(audio_data):
    print("In Loop")
    p = pyaudio.PyAudio()

    # Set up the audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Stream the audio data and append it to the array
    while not keyboard.is_pressed(EXIT_KEY):
        chunk = stream.read(CHUNK)
        audio_data.put(np.frombuffer(chunk, dtype=np.int16))

# Create an empty array to store the audio samples
audio_data = queue.Queue()

# Start the audio streaming in a separate thread
thread = threading.Thread(target=stream_audio, args=(audio_data,))
thread.start()
#=============

#=============
time.sleep(.5)

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


print("Repeat Audio:")
p = pyaudio.PyAudio()
outputstream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output_device_index=OUTPUT_DEVICE,
                output=True,start=False)

outputstream.start_stream()

while audio_data.empty()==False:
    audiobit = audio_data.get()

    seg = AudioSegment(
        audiobit.tobytes(), 
        frame_rate=RATE,
        sample_width=pyaudio.get_sample_size(FORMAT), 
        channels=CHANNELS
    )
    for chunk in make_chunks(seg,CHUNK):
        outputstream.write(chunk._data)

outputstream.stop_stream()
keyboard.wait(EXIT_KEY)
print("Program Exit.")
