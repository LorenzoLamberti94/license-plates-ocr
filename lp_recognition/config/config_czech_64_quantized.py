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
from lpr.trainer import LPRVocab

input_shape = (24, 94, 3)  # (height, width, channels)
use_h_concat = False       # Get all hieroglyphs from train and test
use_oi_concat = False      # Function for treating O/0, I/1 as 1 class
max_lp_length = 20         # Should be at least twice the number of chars in the license plate
rnn_cells_num = 64

convert_to_gray=False
quantization_aware=True

# Licens plate patterns. NB: used just when calculating accuracy. If the pattern is not correct then returns an error "GT label fails ...."
lpr_patterns = [
  '^<[^>]*>[A-Z][0-9A-Z]{5}$', # Oss: the typical prediction looks like this '<Henan>K0F755'
  '^<[^>]*>[A-Z][0-9A-Z][0-9]{3}<police>$',
  '^<[^>]*>[A-Z][0-9A-Z]{4}<[^>]*>$',  # <Guangdong>, <Hebei>
  '^WJ<[^>]*>[0-9]{4}[0-9A-Z]$',
  '^[A-Z]{2}[0-9]{3}[A-Z]{2}$', # Italian
  '^[A-Z0-9]{3}[0-9]{4}$'       # Czech
]
# NB: The caret in the character class ([^) means match anything but >
# Regular expressions tutorial: https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285


# Path to the folder where all training and evaluation artifacts will be located
model_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


class train:
  # Path to annotation file with training data in per line format: <path_to_image_with_license_plate label>
  file_list_path = '/home/lamberti/work/Dataset/Cars_czech/train_czech'

  batch_size = 32
  steps = 250000
  learning_rate = 0.001
  grad_noise_scale = 0.001
  opt_type = 'Adam'

  save_checkpoints_steps = 10000      # Number of training steps when checkpoint should be saved
  display_iter = 100

  apply_basic_aug = False
  apply_stn_aug = True
  apply_blur_aug = False

  need_to_save_weights = True
  need_to_save_log = True

  class execution:
    CUDA_VISIBLE_DEVICES = "1"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class eval:
  # Path to annotation file with validation data in sper line format: <path_to_image_with_license_plate label>
  file_list_path = '/home/lamberti/work/Dataset/Cars_czech/valid_czech'
  checkpoint = '/home/lamberti/work/2_steps_license_plates/lp_recognition/pretrained_model/LPRNet_czech_no_tile_64_quantized/model.ckpt-250000.ckpt'  
  batch_size = 1

  class execution:
    CUDA_VISIBLE_DEVICES = "1"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class infer:
  # Path to text file with list of images in per line format: <path_to_image_with_license_plate>
  file_list_path = '/home/lamberti/work/my_dataset/czech'
  checkpoint = '/home/lamberti/work/2_steps_license_plates/lp_recognition/pretrained_model/LPRNet_czech_no_tile_64_quantized/model.ckpt-250000.ckpt'  
  batch_size = 1

  #Output path where prediction images are saved
  output_dir='/home/lamberti/work/output_debug/'

  class execution:
    CUDA_VISIBLE_DEVICES = "1"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


vocab, r_vocab, num_classes = LPRVocab.create_vocab(train.file_list_path,
                                                    eval.file_list_path,
                                                    use_h_concat,
                                                    use_oi_concat)
