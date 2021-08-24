import pyaudio
import numpy as np

def main():

    CHUNK = 1300
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = np.empty(15600, dtype=np.float32)

    for i in range(12):
        data = stream.read(CHUNK)
        np.append(frames, data)

    print(frames.size)
    print(frames)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
  main()

