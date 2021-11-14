#loading dataset provided Udacity 
from urllib.request import urlretrieve
import os
import cv2
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile


#change directory to where Udacity recorded data is saved
import csv
os.chdir("/opt/carnd_p3/data")
print("Directory changed to udacity data directory")

def uncompress_data(dir,name):
    if(os.path.isdir(name)):
        print('Data extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall('data')
            
uncompress_data('data.zip','data')


# append all entries present in the csv file 
driving_log = [] #simple array to append all the entries present in the .csv file

with open('./data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #to skip the first record as it has the headings
    for line in reader:
        driving_log.append(line) 

        
### Extra Code to append samples from manually created data - write here 

# os.chdir("/opt/carnd_p3/my_data")
# print("Directory changed to my data directory")

# with open('./driving_log.csv') as csvfile: #currently after extracting the file is present in this path
#     reader = csv.reader(csvfile)
#     next(reader, None) #to skip the first record as it has the headings
#     for line in reader:
#         driving_log.append(line) 
        
###end code to gather extra data 

# print("move data from driving log to an array")
# lines = []
# images = []
# measurements = []
#load only image names and steering angle measurements
# for line in lines:    
#     source_path = line[0]
#     tokens = source_path.split('/')
#     print(tokens)
#     filename = tokens[-1]
#     local_path = "./data/"+filename
#     print(local_path)

#     image = cv2.imread(local_path)
#     images.append(image)
#     measurement = line[3]
#     measurements.append(measurement)

#print(len(images))
#print(len(measurements))


# create training and validation datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

training_samples, validation_samples = train_test_split(driving_log,test_size=0.15)
print("training and validation samples created")

#load images 
# function to load image from a row or index
def load_image(index, sample):
    return cv2.imread('IMG/' + sample[index].split('/')[-1])

# function to flip image
def flip_image(image, angle):
    processed_image = cv2.flip(image,1)
    processed_angle = angle*-1.0
    return (processed_image, processed_angle)

#image generator 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:

                # load center image / angle
                center_image = load_image(0, batch_sample) 
                center_angle = float(batch_sample[3])
                # flip center image / angle
                center_flipped = flip_image(center_image, center_angle)
                images.extend([center_image, center_flipped[0]])
                angles.extend([center_angle, center_flipped[1]])
              
                # load left image / angle
                left_image = load_image(1, batch_sample)
                left_angle = center_angle + correction
                # flip left image /angle 
                left_flipped = flip_image(left_image, left_angle)
                images.extend([left_image, left_flipped[0]])
                angles.extend([left_angle, left_flipped[1]])

                # load right image / angle
                right_image = load_image(2, batch_sample)
                right_angle = center_angle - correction
                # load right image / angle
                right_flipped = flip_image(right_image, right_angle)
                images.extend([right_image, right_flipped[0]])
                angles.extend([right_angle, right_flipped[1]])

            X_train = np.array(images) # X_train and y_train are only used to hold the values while generator function is running
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# Train the model using the generator function
training_generator = generator(training_samples, batch_size=32) # batch size # tuning parameter 3 
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, MaxPooling2D

keep_prob = 0.5 #tuning parameter 1
num_epochs = 5 # numbe of epochs #tuning parameter 2 

model = Sequential()
# Normalize image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop image to only show road portion of the image 
model.add(Cropping2D(cropping=((70,25),(0,0))))
# NVIDA model
# Convolution 5x5 Layers 
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu')) # filters = 24, filter size 5X5, stride = 2X2
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu')) # filters = 36, filter size 5X5, stride = 2X2
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu')) # filters = 48, filter size 5X5, stride = 2X2
# Convolution 3x3 Layers
model.add(Conv2D(64,3,3,activation='relu')) # filters = 64, filter size 3x3, stride 1x1
model.add(Conv2D(64,3,3,activation='relu')) # filters = 64, filter size 3x3, stride 1x1
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(keep_prob))
model.add(Flatten())
# Full-Connected Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# output of the model is predicted steering angle 


#using model.fit() history object to produce visulization (taken from lesson)
from keras.models import Model
import matplotlib.pyplot as plt

model.compile(loss='mse', optimizer='adam') # function =mean squared error loss , optimizer - Adam
history_object = model.fit_generator(training_generator, samples_per_epoch=len(training_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=num_epochs, verbose=1) 
# output = steering angle 
model.save('model.h5')
print('Model saved ')

### print the keys contained in the history object
print(history_object.history.keys())
### print model summary
model.summary() 
