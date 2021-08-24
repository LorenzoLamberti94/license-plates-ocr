#!/usr/bin/env openvino
# Run with tensorflow 1.XX (1.15 recommended)
'''
TensorflowLite Model (.tflite) testing script for license plate recognition with LPRNet

WARNING: This is a very experimental script, still needs to be tweaked by hand

Description:
  This script generates the string prediction of the chars of a license plate. 
  It's similar to "infer_frozen_graph_single_image.py" script, but we use now a 
  tflite model (.tflite) instead of a frozen graph (.pb)

Why to use this script:
  It's useful to verify that the tflite_convert command correctly converted 
  a frozen graph to a tflite model( .pb->.tflite )

Input:
    - char_dict: is the {char:indices} vocabulary
    - model_path: path to the TensorflowLite model (.tflite)
    - image_file: path to the testing image

Output to terminal:
  1) predicted string

'''


import numpy as np
import tensorflow as tf
import json
from PIL import Image
import cv2
print(tf.__version__)

def greedy_search_decoder_python(predictions, merge_repeated=True):
    blank_index = (71 - 1)
    output_sequence = list()
    # find max for each row in predictions
    for prediction in predictions:
        output_sequence.append(np.argmax(prediction))
    # create iterators for output_sequence
    output_sequence_iter=output_sequence.copy()
    prev_class_ix = -1
    #remove double predictions and blank spaces
    for prediction in output_sequence_iter:
        if prediction == blank_index or (merge_repeated and prev_class_ix==prediction): 
            output_sequence.remove(prediction)
        prev_class_ix=prediction
    return output_sequence


if __name__ == '__main__':

    # char vocabulary with {char:index} correspondance
    char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<Anhui>': 10, '<Beijing>': 11, '<Chongqing>': 12, '<Fujian>': 13, '<Gansu>': 14, '<Guangdong>': 15, '<Guangxi>': 16, '<Guizhou>': 17, '<Hainan>': 18, '<Hebei>': 19, '<Heilongjiang>': 20, '<Henan>': 21, '<HongKong>': 22, '<Hubei>': 23, '<Hunan>': 24, '<InnerMongolia>': 25, '<Jiangsu>': 26, '<Jiangxi>': 27, '<Jilin>': 28, '<Liaoning>': 29, '<Macau>': 30, '<Ningxia>': 31, '<Qinghai>': 32, '<Shaanxi>': 33, '<Shandong>': 34, '<Shanghai>': 35, '<Shanxi>': 36, '<Sichuan>': 37, '<Tianjin>': 38, '<Tibet>': 39, '<Xinjiang>': 40, '<Yunnan>': 41, '<Zhejiang>': 42, '<police>': 43, 'A': 44, 'B': 45, 'C': 46, 'D': 47, 'E': 48, 'F': 49, 'G': 50, 'H': 51, 'I': 52, 'J': 53, 'K': 54, 'L': 55, 'M': 56, 'N': 57, 'O': 58, 'P': 59, 'Q': 60, 'R': 61, 'S': 62, 'T': 63, 'U': 64, 'V': 65, 'W': 66, 'X': 67, 'Y': 68, 'Z': 69, '_': 70}

    # Define testing image file
    image_file = r"/home/lamberti/work/2_steps_license_plates/dataset/generated_crops_my_dataset_china/0m_1.jpg"

    # Load TFLite model and allocate tensors.
    model_path = "/home/lamberti/work/2_steps_license_plates/lp_recognition/frozen_graph/LPRNet_synth_no_tile_64_quantized/graph.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    first_layer = tensor_details[1]


    # Print infos
    print('image file: ',image_file) #testing image
    print('model pat: ',model_path)  # model loaded

    print("Input details: ", input_details)
    print("Output details: ", output_details)
    print("out first layer: ",first_layer)

    # PILLOW: Open, resize and reshape image
    img_in = Image.open(image_file)
    img_in = img_in.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_array = np.array(img_in, dtype=np.uint8)
    input_array = np.reshape(input_array, input_details[0]['shape'])
    print('np size: ', input_array.shape)
    #cv2.imwrite('/home/lamberti/work/output_debug/image_tflite.png',(input_array).astype(np.uint8)[0]) #debug: print loaded image

    # Test model on input data (process image through the network)
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()


    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    input_data = interpreter.get_tensor(input_details[0]['index'])
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data=output_data[0] # get rid of a dimension

    # decode the output with a greedy decoder. You get the final predicted indices that correspond to chars in the char_dict vocabulary.
    out_char_codes= greedy_search_decoder_python(output_data)

    out_char = []
    for i, char_code in enumerate(out_char_codes):
        for k, v in char_dict.items():
            if char_code == v:
                out_char.append(k)
                continue
    print(out_char_codes) # char codes. indices for the chars in the vocabulary (char_dict)
    print(out_char)       #predicted string

    print('end')
