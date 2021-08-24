# IMPORTANT: This script can be run with either Tensorflow 1.XX and Tensorflow 2.X,
# but with Tensorflow 2.X is much faster!

import os
import pathlib
import sys

# Add Tensorflow Object Detection API "models" directory to import libraries: https://github.com/tensorflow/models/
sys.path.append('../external/tensorflow-api/research/')

#general import
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print(tf.__version__)

import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

# Tensorflow Object Detection API modules
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

################################################################################


import argparse
def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    parser.add_argument('-d', '--data_path', help='path to dataset you want to test on',
                        default='../dataset/my_dataset/')
    parser.add_argument('-m', '--model', help='path to dataset you want to test on',
                        default='./pre-trained/output_inference_graph_320x240_ssdlitev2_8bit_sym.pb')
    parser.add_argument('-o', '--output_path', help='folder where you want to save your detections',
                        default='./output/')
    parser.add_argument('-t', '--threshold', help='threshold for detection probability in [0.0 , 1.0] range',
                        default=0.30,  type=float)
    parser.add_argument('--label_map', help='path to the label map file (.pbtxt)',
                        default='../dataset/oid_v4_label_map_license_plates.pbtxt')
    parser.add_argument('--draw_detection_box', action='store_true', 
                        help='If you activate this, the output images will be saved in the original \
                          size and the detection will be drawn over the image. It this is deactivated, \
                          then the images will be saved cropped on the bounding box size')                           
    return parser

################################################################################


## Model Loader
def load_model(model_dir):
  model_dir = pathlib.Path(model_dir)/"saved_model"
  
  if tf.__version__[0] == '1':
    print('loading model with Tensorflow version 1.XX')
    model = tf.compat.v2.saved_model.load(str(model_dir), None) #TF 1.15
  if tf.__version__[0] == '2':
    print('loading model with Tensorflow version 2.XX')
    model = tf.saved_model.load(str(model_dir)) # TF 2.1

  model = model.signatures['serving_default']

  return model


def get_image_list(directory_path):
  #convert string to path
  directory_path = pathlib.Path(directory_path)
  # define which extensions you want to search for 
  extensions = ("*.png","*.jpg","*.jpeg",)

  #create a list with all the paths of the images inside the "directory_path"
  images_paths = []
  for extension in extensions:
      images_paths.extend(directory_path.glob(extension))
  # sort by name    
  images_paths = sorted(images_paths)
  return images_paths


## Add a wrapper function to call the model, and cleanup the outputs:
def run_inference_for_single_image(model, image):
  # All outputs in "output_dict" are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.

  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]


  # Generate Output Tensors
  output_dict = model(input_tensor)
 
  # Run inference TF1.XX
  if tf.__version__[0] == '1':
    with tf.Session() as sess:
      print('TF 1.XX is being used. Session opened, wait for evaluation...')
      # num_detections=sess.run(output_dict['num_detections'])
      # num_detections=output_dict['num_detections'].eval()
      for key,value in output_dict.items():
        output_dict[key] = value.eval()
        output_dict[key] = output_dict[key][0] # take index [0] to remove the batch dimension 
    num_detections = int(output_dict['num_detections'])
    print('Evaluation completed.')

  # Run inference TF2.XX
  elif tf.__version__[0] == '2':
    print('TF 2.XX is being used. Wait for evaluation...')
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    print('Evaluation completed.')

  else:
    print('check your tensorflow version. TF2.1 or TF1.15 recommended')
    exit

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def prune_output(input_dict, min_score_thresh, max_boxes_to_draw=10):
  ''' prune all the outputs that are under a threshold ("min_score_thresh")'''

  ''' (TODO) Alternative with np arrays: (to be finished)
  max_boxes_to_draw = min(max_boxes_to_draw, input_dict['num_detections'])

  boxes = input_dict['detection_boxes'][:max_boxes_to_draw]
  classes = input_dict['detection_classes'][:max_boxes_to_draw]
  scores = input_dict['detection_scores'][:max_boxes_to_draw]

  indices=np.argwhere(scores > min_score_thresh)[0]
  boxes = input_dict['detection_boxes'][indices]
  classes = input_dict['detection_classes'][indices]
  scores = input_dict['detection_scores'][indices]
  output_dict={}
  '''

  # define output structure
  output_dict={ 'detection_boxes':  [],
                'detection_classes': [], 
                'detection_scores':  []
                }

  boxes = input_dict['detection_boxes']
  classes = input_dict['detection_classes']
  scores = input_dict['detection_scores']

  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      # box = tuple(boxes[i].tolist())
      output_dict['detection_boxes'].append(boxes[i].tolist())
      output_dict['detection_classes'].append(classes[i].tolist())
      output_dict['detection_scores'].append(scores[i].tolist())

  return output_dict

def perform_inference(model, image_path):
  image_np = np.array(Image.open(image_path).convert("RGB"))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  return output_dict


def overlap_images_with_bboxes(output_dict, image_path, category_index, thresh=1.0):
  # the array based representation of the image will be used in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path).convert("RGB"))
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      min_score_thresh=thresh,
      max_boxes_to_draw=3,
      line_thickness=2)
  return image_np

def generate_crops(output_dict, image_path):
  # the array based representation of the image will be used in order to prepare the result image crop 
  image_np = np.array(Image.open(image_path).convert("RGB"))
  # get size of the image
  h, w = image_np.shape[0], image_np.shape[1]

  if len(output_dict['detection_scores']) >1:
    print('debug')

  image_multiple_crops=[] #if an image has more than 1 license plate, then store all of them!
  if output_dict['detection_boxes']:
    for bbox in output_dict['detection_boxes']:
      #take bbox coordinates
      ymin, xmin, ymax, xmax = bbox
      #denormalize coordinates
      ymin, xmin, ymax, xmax = int(ymin*h), int(xmin*w), int(ymax*h), int(xmax*w)
      #crop the image
      image_crop = image_np[ ymin:ymax , xmin:xmax ]
      # put the crop in a list 
      image_multiple_crops.append(image_crop)
  else:
      print('\n\nPlate not detected in:', image_path.name )
  
  return image_multiple_crops

def save_np_image(np_img, image_saving_path, image_name):
  #convert strings in paths
  image_saving_path=pathlib.Path(image_saving_path)
  image_name=pathlib.Path(image_name)

  # convert to pillow format
  image_pillow = Image.fromarray(np_img)
  ## Save Image
  image_pillow.save(image_saving_path.joinpath(image_name))


################################################################################
# MAIN
################################################################################

if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  ## Threshold for detections in [0,100] range
  THRESH = args.threshold

  ## Images To Test On
  PATH_TO_TEST_IMAGES_DIR = args.data_path

  ## Output folder for inference
  image_saving_path = args.output_path

  ## Loading Label Map: List of the strings that is used to add correct label for each box.
  PATH_TO_LABEL_MAP = args.label_map
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_MAP, use_display_name=True)

  ## create a list with all images paths
  TEST_IMAGE_PATHS = get_image_list(PATH_TO_TEST_IMAGES_DIR) 
  print(len(TEST_IMAGE_PATHS),"Testing images found: \n")
  print(*TEST_IMAGE_PATHS, sep = "\n")

  ## Set Model
  model_dir  = args.model
  detection_model = load_model(model_dir)

  ## print model info
  print('\nModel infos:')
  print('inputs: ', detection_model.inputs)
  print('outputs: ', detection_model.output_dtypes)
  print('output shape: ', detection_model.output_shapes)


  # create output folder if it does not exists
  if not os.path.exists(image_saving_path):
      os.mkdir(image_saving_path)
      print("Directory", image_saving_path, "created")
  else:    
      print("Directory", image_saving_path, "already exists")



  ## Run on each test image and show the results:
  for image_path in TEST_IMAGE_PATHS:

    all_predictions_dict = perform_inference(detection_model, image_path)
    # Clean predictions dictionary and apply threshold on detections
    output_dict = prune_output(all_predictions_dict, THRESH)
    print(score for score in output_dict['detection_scores'])

    ## Save LP original images with bbox detections
    image_with_detections = overlap_images_with_bboxes(all_predictions_dict, image_path, category_index, thresh=THRESH) # automatically applies threshold
    if args.draw_detection_box:
      save_np_image(image_with_detections, image_saving_path, image_path.name)
      # display(Image.fromarray(image_np))
    else:
      # Save LP cropped images
      lp_crops = generate_crops(output_dict, image_path) 
      for crop in lp_crops:
        save_np_image(crop, image_saving_path, image_path.name)
        # display(Image.fromarray(image_np))
        break # save just first crop, with the highest score

  print('end of the script.')