import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import tflite_runtime.interpreter as tflite


def main():

    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, 'lite-model_yamnet_classification_tflite1.tflite')
    label_file = os.path.join(script_dir, 'AudioLabels.txt')
    image_file = os.path.join(script_dir, 'test1.png')

    # Initialize the TF interpreter
    interpreter = tflite.Interpreter(model_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    #interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()

    # Resize the image
   # size, height = common.input_size(interpreter)
    image = Image.open(image_file)

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




