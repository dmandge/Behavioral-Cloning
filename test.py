#loading dataset provided Udacity 
from urllib.request import urlretrieve
import os
import cv2
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile
from PIL import Image 

#change directory to where Udacity recorded data is saved
import csv
# os.chdir("/opt/carnd_p3/data")
# print("Directory changed to udacity data directory")

# def uncompress_data(dir,name):
#     if(os.path.isdir(name)):
#         print('Data extracted')
#     else:
#         with ZipFile(dir) as zipf:
#             zipf.extractall('data')
            
# uncompress_data('data.zip','data')

def flip_image(image):
    processed_image = cv2.flip(image,1)
    #processed_angle = angle*-1.0
    return (processed_image)

image = img = cv2.imread('2021_04_18_16_31_45_852.jpg')
save_image = flip_image(image)
cv2.imwrite('flip.jpg', save_image)

             


