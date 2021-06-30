"""Keyword spotter model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import audio_recorder
import numpy as np
import queue
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

q = queue.Queue()

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)-8s %(asctime)-15s %(name)s %(message)s")
audio_recorder.logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def get_queue():
  return q

def classify_audio(audio_device_index, interpreter, labels_file,
                   commands_file=None,
                   result_callback=None, dectection_callback=None,
                   sample_rate_hz=16000,
                   negative_threshold=0.6, num_frames_hop=33):
  """Acquire audio, preprocess, and classify."""
  # Initialize recorder.
  AUDIO_SAMPLE_RATE_HZ = sample_rate_hz
  downsample_factor = 1
  if AUDIO_SAMPLE_RATE_HZ == 48000:
    downsample_factor = 3
  # Most microphones support this
  # Because the model expects 16KHz audio, we downsample 3 fold
  recorder = audio_recorder.AudioRecorder(
      AUDIO_SAMPLE_RATE_HZ,
      downsample_factor=downsample_factor,
      device_index=audio_device_index)
  labels = read_labels(labels_file)
  if commands_file:
      commands = read_commands(commands_file)
  else:
      commands = {}
  logger.info("Loaded commands: %s", str(commands))
  logger.info("Recording")
  timed_out = False
  with recorder:
      last_detection = -1
      while not timed_out:
          audio = recorder.get_audio(15600)
          set_input(interpreter, audio)
          interpreter.invoke()












def input_tensor(interpreter):
    """Returns the input tensor view as numpy array."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to input tensor."""
    interpreter_shape = interpreter.get_input_details()[0]['shape']
    input_tensor(interpreter)[:,:] = np.reshape(data, interpreter_shape[1:3])


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def get_output(interpreter):
    """Returns entire output, threshold is applied later."""
    return output_tensor(interpreter, 0)

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)

def read_labels(filename):
  # The labels file can be made something like this.
  f = open(filename, "r")
  lines = f.readlines()
  return ['negative'] + [l.rstrip() for l in lines]


def read_commands(filename):
  # commands should consist of a label, a command and a confidence.
  f = open(filename, "r")
  commands = {}
  lines = f.readlines()
  for command, key, confidence in [l.rstrip().split(',') for l in lines]:
    commands[command] = { 'key': key, 'conf': 0.4}
    if confidence and 0 <= float(confidence) <= 1:
      commands[command]['conf'] = float(confidence)
  return commands

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def add_model_flags(parser):
  parser.add_argument(
      "--model_file",
      help="File path of TFlite model.",
      default="coral.tflite")
  parser.add_argument("--mic", default=None,
                      help="Optional: Input source microphone ID.")
  parser.add_argument(
      "--num_frames_hop",
      default=33,
      help="Optional: Number of frames to wait between model inference "
      "calls. Smaller numbers will reduce the latancy while increasing "
      "compute cost. Must devide 198. Defaults to 33.")
  parser.add_argument(
      "--sample_rate_hz",
      default=16000,
      help="Optional: Sample Rate. The model expects 16000. "
      "However you may alternative sampling rate that may or may not work."
      "If you specify 48000 it will be downsampled to 16000.")
