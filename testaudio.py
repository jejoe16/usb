import pyaudio
import numpy as np
import samplerate as sr

def main():

    CHUNK = 15600
    RECORD_SECONDS = 5
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 48000
    TARGETRATE = 16000

    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print((i, dev['name'], dev['maxInputChannels']))


    defaultCapability = p.get_default_host_api_info()
    print(defaultCapability)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=0)

    frames = np.empty(15600, dtype=np.float32)

    ratio = TARGETRATE / RATE
    resampler = sr.Resampler()

    raw_data = stream.read(15600)
    data = np.fromstring(frames, dtype=np.int16)
    resampled_data = resampler.process(data, ratio)
    print('{} -> {}'.format(len(data), len(resampled_data)))

    data = stream.read(CHUNK,exception_on_overflow = False)
    np.append(frames, data)

    # Output as a WAV file
    import scipy.io.wavfile as wav
    wav.write('out.wav', RATE, frames)


    print(frames.size)
    print(frames)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
  main()

