from keras.models import Model as KerasModel
from keras.layers import Input, Dense,Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
#from keras.optimizers import Adam
import os
import glob
import torch
import cv2
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
IMGWIDTH = 256
import face_recognition
import torch
import numpy as np
# Get all test videos
#filename = 'ahjnxtiamx.mp4'
import cv2
# Number of frames to sample (evenly spaced) from each video
n_frames = 10


class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        self.model.compile(optimizer = 'adam',loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3,3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2,2), padding='same')(x1)
        
        x2 = Conv2D(8,(5,5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2,2), padding='same')(x2)
        
        x3 = Conv2D(16, (5,5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2,2), padding='same')(x3)
        
        x4 = Conv2D(16,(5,5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4,4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        
        return KerasModel(inputs=x, outputs=y)
from keras.preprocessing.image import ImageDataGenerator

def run_model(filename):

    MesoNet_classifier = Meso4()
    MesoNet_classifier.load("Meso4_DF")


    with torch.no_grad():
        face_list_whole_data=[]
        if True:
            try:
                # Create video reader and find length
                v_cap = cv2.VideoCapture(filename)
                v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Pick 'n_frames' evenly spaced frames to sample
                sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)
                face_list_1video = []
                for j in range(v_len):
                    if j in sample:
                        success, vframe = v_cap.read()
                        vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(vframe)
                        face_list_1frame=[]
                        for face_location in face_locations:
                        # Print the location of each face in this image
                            top, right, bottom, left = face_location
                        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                        # Access the actual face itself:
                            face_image = vframe[top:bottom, left:right]
                            res = cv2.resize(face_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                            face_list_1frame.append(res/255)
                        face_list_1video.append(face_list_1frame)
                face_list_whole_data.append(face_list_1video)
            except KeyboardInterrupt:
                raise Exception("Stopped.")

    print(face_list_whole_data)


    all_prob_list=[0]*len(face_list_whole_data)
    print(face_list_whole_data)
    for i, video in enumerate(face_list_whole_data):
        prob_list=[]
        for j, frame in enumerate(video):
            img_array=np.array(frame)
            probabilistic_predictions = MesoNet_classifier.predict(img_array)
            prob_list.append(probabilistic_predictions)
        all_prob_list[i]=prob_list
            #predictions = [num_to_label[round(x[0])] for x in probabilistic_predictions]
            #print(predictions)



    bias = -0.4
    weight = 0.068235746
    print(all_prob_list)
    submission = []
    subm_prob=[]
    for fn, prob in zip(filename, all_prob_list):
        print(fn)
        if prob is not None and len(prob) == 10:
            indiv_prob=[]
            for i in prob:
                indiv_prob.append(i)
        subm_prob.append(indiv_prob)
        submission.append([os.path.basename(fn), sum(indiv_prob)/len(indiv_prob)])

    return(int(submission[0][1][0]+.5))
