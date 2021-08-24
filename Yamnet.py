import os
import numpy as np
import pathlib
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import tflite_runtime.interpreter as tflite


def main():

    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, 'lite-model_yamnet_classification_tflite.tflite')
    label_file = os.path.join(script_dir, 'AudioLabels.txt')
    image_file = os.path.join(script_dir, 'test1.png')

    # Initialize the TF interpreter
    interpreter = tflite.Interpreter(model_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    #interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()

    # Get Audio 15600 frames float32[15600] - Input audio clip to be classified (16 kHz float32 waveform samples in range -1.0..1.0).
    image = np.zeros(int(round(0.975 * 16000)), dtype=np.int32)
    print(image.shape)  # Should print (15600,)

    input_details = interpreter.get_input_details()
    print(input_details)

    #print(interpreter.resize_tensor_input(interpreter, [15600], strict=True))

    #print(common.input_tensor(interpreter))

    # Run an inference
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))



if __name__ == "__main__":
  main()




