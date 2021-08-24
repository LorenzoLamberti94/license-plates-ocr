'''
This script takes a folder path, lists all the images inside that folder and
creates a .txt file in which it writes on each row the path of the images in that folder 
It ignores Folders in the path given.
It doesnt list images on subfolders

WHY DO YOU NEED THIS Script:
LPRNet has testing script that needs a txt file as input to locate all the testing 
images. Each row represent the path of an image that will be tested  

inputs:
   path_to_images: is the path to the folder ofthe images that you want to list
   output_file_name: the name of the output file i.e. "image_list.txt".

output:
    a txt file written named output_file_name to the current script directory

'''

import os
from os.path import join

path_to_images = '/home/lamberti/work/my_dataset/my_dataset'
output_file_name = 'image_list.txt'

items = sorted(os.listdir(path_to_images)) #this gives me a list of both files and folders in dir

current_dir_path = os.path.dirname(os.path.realpath(__file__))

# Open the file in append & read mode ('a+')
with open( join(current_dir_path , output_file_name), "a+") as file_object:
    
    for name in items:
        if os.path.isfile(path_to_images + name) == True:
            print('valid file:' ,path_to_images + name)
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")

            file_object.write(path_to_images + name)
        else:
            print('not valid file:' , path_to_images + name)

print('All images have been written in the following file: \n', join(current_dir_path , output_file_name))

