#!/usr/bin/env openvino
# Run with tensorflow 1.XX (1.15 recommended)
'''
Testing script for license plate recognition with LPRNet

This script generates images with prediction superimposed. 
It does not calculate the predictions accuracy. Use eval_lpr.py for that.

Detailed description:
  The script takes a trained model (.ckpt file) (config.infer.checkpoint) and 
  tests the network on the dataset specified in the configuration file 
  (config.infer.file_list_path). 

Input:
    - path_to_config: path to the configuration file

Output:
  1) In the terminal you will see only which images are being tested.
  2) At the end all the images tested will be saved with the corresponding prediction 
     at the path specified in the config file (config.infer.output_dir)

Example Command:
  python test_lpr.py config/config_file.py 
'''



import os
import pathlib
import sys
# Set working directory
os.chdir('/home/lamberti/work/2_steps_license_plates/lp_recognition') 
# Add openvino training extensions "lpr" directory to import libraries: https://github.com/opencv/openvino_training_extensions/tree/develop/tensorflow_toolkit/lpr
sys.path.append('/home/lamberti/work/openvino_training_extensions/tensorflow_toolkit/lpr')
sys.path.append('/home/lamberti/work/openvino_training_extensions/tensorflow_toolkit/utils') # solves "tfutils" not found error

import argparse
import random
import numpy as np
import cv2
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
print(tf.__version__)
from lpr.utils import dataset_size
from lpr.trainer import inference, decode_beams
from tfutils.helpers import load_module
from tools import debug # my own debug script

### Configuration file path
# PATH_TO_CONFIG_FILE = '/home/lamberti/work/2_steps_license_plates/lp_recognition/config/config_czech_gray.py' # my European dataset grayscale
# PATH_TO_CONFIG_FILE = '/home/lamberti/work/2_steps_license_plates/lp_recognition/config/config_czech.py' # my European dataset
# PATH_TO_CONFIG_FILE = '/home/lamberti/work/2_steps_license_plates/lp_recognition/config/config_synthetic.py' # synth dataset

def parse_args():
  parser = argparse.ArgumentParser(description='Infer of a trained model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def read_data(height, width, channels_num, list_file_name, batch_size=1, convert_to_gray=False):
  reader = tf.TextLineReader()
  _, value = reader.read(list_file_name)
  filename = value
  image_filename = tf.read_file(filename)
  image = tf.image.decode_png(image_filename, channels=channels_num)
  if convert_to_gray:
    image = tf.image.rgb_to_grayscale(image, name=None)  # have a ?,?,1 channel gray image
    image = tf.tile(image, [1, 1, 3])                    # have a ?,?,3 channel gray image
  image_float = tf.image.convert_image_dtype(image, tf.float32)
  resized_image = tf.image.resize_images(image_float, [height, width])
  resized_image.set_shape([height, width, channels_num])

  image_batch, file_batch = tf.train.batch([resized_image, filename], batch_size=batch_size,
                                           allow_smaller_final_batch=True)
  return image_batch, file_batch

def data_input(height, width, channels_num, filename, batch_size=1, convert_to_gray=False):
  files_string_producer = tf.train.string_input_producer([filename])
  image, filename = read_data(height, width, channels_num, files_string_producer, batch_size)
  return image, filename


def get_image_list(PATH_TO_TEST_IMAGES_DIR):
  # PATH_TO_TEST_IMAGES_DIR = pathlib.Path(PATH_TO_TEST_IMAGES_DIR)
  extensions = ("*.png","*.jpg","*.jpeg",)
  paths = []
  for extension in extensions:
      paths.extend(glob.glob( os.path.join(PATH_TO_TEST_IMAGES_DIR,extension) ))
  TEST_IMAGE_PATHS = sorted(paths)
  return TEST_IMAGE_PATHS

def read_data_from_list(height, width, channels_num, filename_queue , batch_size=1, convert_to_gray=False):
  # step 3: read, decode and resize images
  reader = tf.WholeFileReader()
  filename, content = reader.read(filename_queue)
  image = tf.image.decode_png(content, channels=channels_num)
  if convert_to_gray:
    image = tf.image.rgb_to_grayscale(image, name=None)  # have a ?,?,1 channel gray image
    image = tf.tile(image, [1, 1, 3])                    # have a ?,?,3 channel gray image
  image_float = tf.image.convert_image_dtype(image, tf.float32)
  resized_image = tf.image.resize_images(image_float, [height, width])            
  resized_image.set_shape([height, width, channels_num])

  # step 4: Batching
  image_batch, file_batch = tf.train.batch([resized_image, filename], batch_size=batch_size,
                                          allow_smaller_final_batch=True)                                           
  return image_batch, file_batch

def data_input_from_list(height, width, channels_num, images_paths, batch_size=1, convert_to_gray=False):
  filename_queue = tf.train.string_input_producer(images_paths)
  image, filename = read_data_from_list(height, width, channels_num, filename_queue, batch_size, convert_to_gray)
  return image, filename

# pylint: disable=too-many-statements, too-many-locals
def infer(config):
  if hasattr(config.infer, 'random_seed'):
    np.random.seed(config.infer.random_seed)
    tf.set_random_seed(config.infer.random_seed)
    random.seed(config.infer.random_seed)

  if hasattr(config.infer.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape
  rnn_cells_num = config.rnn_cells_num

  # DEBUGGING FUNCTION
  debug.run_sessions2(config) # Compare TF function with my own implementation to check if they are the same

  graph = tf.Graph()

  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      # Read images list from txt file 
      inp_data, filenames = data_input(height, width, channels_num, config.infer.file_list_path,
                                       batch_size=config.infer.batch_size,
                                       convert_to_gray=config.convert_to_gray)
      # Read images list from folder path 
      # images_paths = get_image_list(config.infer.file_list_path)
      # inp_data, filenames = data_input_from_list(height, width, channels_num, images_paths,
      #                                   batch_size=config.infer.batch_size, 
      #                                   convert_to_gray=config.convert_to_gray)
      
      prob = inference(rnn_cells_num, inp_data, config.num_classes)
      prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC
      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size

      # perform greedy decoding
      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)

      predictions = tf.to_int32(result[0][0])
      d_predictions = tf.sparse_to_dense(predictions.indices,
                                          [tf.shape(inp_data, out_type=tf.int64)[0], config.max_lp_length],
                                          predictions.values, default_value=-1, name='d_predictions')

      # quantization aware training
      if config.quantization_aware:
        tf.contrib.quantize.experimental_create_eval_graph(input_graph=graph, symmetric=True)

      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  # session
  conf = tf.ConfigProto()
  if hasattr(config.eval.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.eval.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  sess = tf.Session(graph=graph, config=conf)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  sess.run(init)

  latest_checkpoint = config.infer.checkpoint
  if config.infer.checkpoint == '':
    latest_checkpoint = tf.train.latest_checkpoint(config.model_dir)

  saver.restore(sess, latest_checkpoint)

  # load from txt file
  infer_size = dataset_size(config.infer.file_list_path)
  # load from folder path
  # infer_size = len(images_paths)

  steps = int(infer_size / config.infer.batch_size) if int(infer_size / config.infer.batch_size) else 1

  for _ in range(steps):

    vals, batch_filenames , image = sess.run([d_predictions, filenames, inp_data])
    print(batch_filenames)
    pred = decode_beams(vals, config.r_vocab)

    for i, filename in enumerate(batch_filenames):
      filename = filename.decode('utf-8')

      ## read image with cv2
      # img = cv2.imread(filename)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ## keep image red with tensorflow
      img = (image*255).astype(int)[i]

      size = cv2.getTextSize(pred[i], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
      text_width = size[0][0]
      text_height = size[0][1]

      img_he, img_wi, _ = img.shape
      img = cv2.copyMakeBorder(img, 0, text_height + 10, 0,
                               0 if text_width < img_wi else text_width - img_wi, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
      cv2.putText(img, pred[i], (0, img_he + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
      image_name = os.path.basename(os.path.normpath(filename))

      cv2.imwrite(config.infer.output_dir + image_name ,img)
      img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) #converto from BGR to RGB
      cv2.imwrite(config.infer.output_dir + image_name ,img)
      # cv2.imshow('License Plate', img)
      # key = cv2.waitKey(0)


  coord.request_stop()
  coord.join(threads)
  sess.close()


def main(_):
  # Decomment this to parse args:
  args = parse_args()
  cfg = load_module(args.path_to_config)

  # Decomment if you don't want to parse args
  # cfg = load_module(PATH_TO_CONFIG_FILE)
  infer(cfg)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
