import os
import numpy as np
import pathlib
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import tflite_runtime.interpreter as tflite
import pyaudio

def main():

    CHUNK = 1024
    RECORD_SECONDS = 5
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=0)

    frames = np.empty(15600, dtype=np.float32)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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



    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, 'lite-model_yamnet_classification_tflite.tflite')
    label_file = os.path.join(script_dir, 'AudioLabels.txt')
    image_file = os.path.join(script_dir, 'test1.png')

    # Initialize the TF interpreter
    interpreter = tflite.Interpreter(model_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

    # Get Audio 15600 frames float32[15600] - Input audio clip to be classified (16 kHz float32 waveform samples in range -1.0..1.0).
    #audio = np.zeros(15600, dtype=np.float32)
    audio = frames
    #print(image.shape)  # Should print (15600,)

    interpreter.resize_tensor_input(0, [15600], strict=True)
    interpreter.allocate_tensors()

    #input_details = interpreter.get_input_details()
    #print(input_details)

    interpreter.set_tensor(0, audio)
    # Run an inference
    #common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))



if __name__ == "__main__":
  main()




