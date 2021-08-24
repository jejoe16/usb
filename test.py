import tensorflow as tf
import numpy as np
import zipfile

# Download the model to yamnet-classification.tflite
interpreter = tf.lite.Interpreter('lite-model_yamnet_classification_tflite.tflite')

input_details = interpreter.get_input_details()
print(input_details)
waveform_input_index = input_details[0]['index']
print(waveform_input_index)
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
print(output_details)

# Input: 0.975 seconds of silence as mono 16 kHz waveform samples.
waveform = np.zeros(15600, dtype=np.float32)
print(waveform.shape)  # Should print (15600,)

interpreter.resize_tensor_input(0, [15600], strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(waveform_input_index, waveform)
interpreter.invoke()
scores = interpreter.get_tensor(scores_output_index)
print(scores.shape)  # Should print (1, 521)

top_class_index = scores.argmax()
labels_file = zipfile.ZipFile('lite-model_yamnet_classification_tflite.tflite').open('yamnet_label_list.txt')
labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]
print(len(labels))  # Should print 521
print(labels[top_class_index])  # Should print 'Silence'.