#!/usr/bin/env openvino
# Run with tensorflow 1.XX (1.15 recommended)
'''
Frozen Model (.pb) testing script for license plate recognition with LPRNet

Description:
  This script generates images with prediction superimposed. 
  It does not calculate the predictions accuracy. Use eval_lpr.py for that.
  It's the exact same script as test_lpr.py, but we use now a frozen graph (.pb)
  instead of a checkpoint (.ckpt).

Why to use this script:
  It's useful to verify that the export_frozen_graph.py script correctly converted 
  a ckeckpoint to frozen graph ( .ckpt->.pb )

Input:
    - path_to_config: path to the configuration file
    - model: path to the frozen graph file (.pb)
    - input_image: path to the testing image
    - output: it's the path and name of the generated output image (format: path/name_image.png). This image has the predicted characters superimposed

Output:
  1) The input_image tested will be saved with the corresponding prediction 
     at the path specified by "--output"

Example Command:
  python infer_frozen_graph_single_image.py --config config/config_file.py --model frozen_graph/LPRNet_synth_no_tile_quantized/graph.pb  --output output_debug/frozen.jpg --input_image /home/lamberti/work/my_dataset/my_dataset_china_crops/0m_1.jpg 

#OBS: config file needed just to build the vocabulary with {index:char} correspondence
'''
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import sys
# Set working directory
os.chdir('/home/lamberti/work/2_steps_license_plates/lp_recognition') 
# Add openvino training extensions "lpr" directory to import libraries: https://github.com/opencv/openvino_training_extensions/tree/develop/tensorflow_toolkit/lpr
sys.path.append('/home/lamberti/work/openvino_training_extensions/tensorflow_toolkit/lpr')
sys.path.append('/home/lamberti/work/openvino_training_extensions/tensorflow_toolkit/utils') # solves "tfutils" not found error

from argparse import ArgumentParser
import numpy as np
import cv2
import tensorflow as tf
print(tf.__version__)

from tfutils.helpers import load_module
from lpr.trainer import decode_beams

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('--model', help='Path to frozen graph file with a trained model.', required=True, type=str)
  parser.add_argument('--config', help='Path to a config.py', required=True, type=str)
  parser.add_argument('--output', help='Output image path/name.png')
  parser.add_argument('--input_image', help='Image with license plate')
  return parser

def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
  return graph

def display_license_plate(number, license_plate_img):
  size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
  text_width = size[0][0]
  text_height = size[0][1]

  height, width, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                         0 if text_width < width else text_width - width,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def main():
  args = build_argparser().parse_args()

  graph = load_graph(args.model)
  config = load_module(args.config)

  image = cv2.imread(args.input_image)
  img = cv2.resize(image, (94, 24))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  img = np.multiply(img, 1.0/255.0)

  input = graph.get_tensor_by_name("import/input:0")
  output = graph.get_tensor_by_name("import/d_predictions:0")

  with tf.Session(graph=graph) as sess:
    results = sess.run(output, feed_dict={input: [img]})
    print(results)

    decoded_lp = decode_beams(results, config.r_vocab)[0]
    print(decoded_lp)

    img_to_display = display_license_plate(decoded_lp, image)

    if args.output:
      cv2.imwrite(args.output, img_to_display)
    # else:
    #   cv2.imshow('License Plate', img_to_display)
    #   cv2.waitKey(0)


if __name__ == "__main__":
  main()
