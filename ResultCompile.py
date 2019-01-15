from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.utils.np_utils import to_categorical
import scipy.io as sio
import h5py
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.models import Sequential, Model
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from collections import Counter

import keras.backend as K
from keras.backend.common import _EPSILON

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))

#ResNet-50 model for present task
def prepareResModel():
    model = ResNet50(weights='imagenet')
    model.layers.pop()
    image = model.input
    model = Dense(1, activation='sigmoid', name='loss')(model.layers[-1].output)
    model = Model(input=image, output=[model])
    return model

#MobileNet model for present task	
def prepareMobileModel():	
	model=MobileNet(weights='imagenet')
	image = model.input
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()
	model.layers.pop()
	model = Dense(1, activation='sigmoid', name='loss')(model.layers[-1].output)
	model = Model(input=image, output=[model])
	return model



algo='CCE_Error'
#'CEMD', 'MSE_Error' 'CCE_Error'
batch_size =4
# .mat file read by h5py.File() possible if the .mat file is saved in v7.3
f = h5py.File('/home/cvpr/Anabik/SeverityExperiment2/Data/Psoriasis_Severity_Data_224.mat','r');
totalImg = f.get('images')
label = f.get('labels')

Size=101;

Ery_score = np.zeros((707,4),'float32')
Sca_score = np.zeros((707,4),'float32')
Ind_score = np.zeros((707,3),'float32')
for fold in range(7):
  for bar in range(4):
    model = prepareResModel(); # Use prepareMobileModel(noc) for Mobile Net
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    model_weight_Path = "Ery_Fold{}_bar{}_Full_Weight.mat".format(fold + 1,bar)
    model.load_weights(model_weight_Path )
    testStartRange=Size*fold
    testEndRange=Size*(fold+1)
    testRange=range(testStartRange,testEndRange)
    testImg=totalImg[testRange,:,:,:]
    testImg=testImg.astype('float32')
    testImg = np.swapaxes(testImg, 1, 3)
    test_score = model.predict(testImg)
    Ery_score[testRange,bar] = test_score[:,0]

for fold in range(7):
  for bar in range(4):
    model = prepareResModel(); # Use prepareMobileModel(noc) for Mobile Net
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    model_weight_Path = "Sca_Fold{}_bar{}_Full_Weight.mat".format(fold + 1,bar)
    model.load_weights(model_weight_Path )
    testStartRange=Size*fold
    testEndRange=Size*(fold+1)
    testRange=range(testStartRange,testEndRange)
    testImg=totalImg[testRange,:,:,:]
    testImg=testImg.astype('float32')
    testImg = np.swapaxes(testImg, 1, 3)
    test_score = model.predict(testImg)
    Sca_score[testRange,bar] = test_score[:,0]


for fold in range(7):
  for bar in range(3):
    model = prepareResModel(); # Use prepareMobileModel(noc) for Mobile Net
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    model_weight_Path = "Ind_Fold{}_bar{}_Full_Weight.mat".format(fold + 1,bar)
    model.load_weights(model_weight_Path )
    testStartRange=Size*fold
    testEndRange=Size*(fold+1)
    testRange=range(testStartRange,testEndRange)
    testImg=totalImg[testRange,:,:,:]
    testImg=testImg.astype('float32')
    testImg = np.swapaxes(testImg, 1, 3)
    test_score = model.predict(testImg)
    Ind_score[testRange,bar] = test_score[:,0]

savePath="Bin_Error_Full.mat"

sio.savemat(savePath, mdict={'Ery_Score': Ery_score,'Sca_Score': Sca_score,'Ind_Score': Ind_score});



