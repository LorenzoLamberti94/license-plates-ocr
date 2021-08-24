# takes a folder (ORIGIN_PATH) full of rgb images and save all these images to a 
# destination path (DESTIN_PATH) in grayscale format.

import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join

## Train
ORIGIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/train/Vehicle_registration_plate/"
DESTIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/grayscale/train/Vehicle_registration_plate/"

# ## Validation
# ORIGIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/validation/Vehicle_registration_plate/"
# DESTIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/grayscale/validation/Vehicle_registration_plate/"

# ## Test
# ORIGIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/test/Vehicle_registration_plate/"
# DESTIN_PATH = "/home/lamberti/work/open_images_v4_dataset/Dataset/grayscale/test/Vehicle_registration_plate/"

try:
    makedirs(DESTIN_PATH)
except:
    print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = [f for f in listdir(ORIGIN_PATH) if isfile(join(ORIGIN_PATH,f))] 

for image in files:
    
    # print(os.path.join(ORIGIN_PATH,image))
    img = cv2.imread(os.path.join(ORIGIN_PATH,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    DESTIN_PATH_JOIN = join(DESTIN_PATH,image)
    print(DESTIN_PATH_JOIN)
    cv2.imwrite(DESTIN_PATH_JOIN,gray)
    # except:
    #     print ("{} is not converted".format(image))

print('all the images have been converted')