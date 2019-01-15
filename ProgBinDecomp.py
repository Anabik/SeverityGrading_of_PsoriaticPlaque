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
from modifaugmentation import ImageDataGenerator
from collections import Counter
import keras.backend as K
from keras.backend.common import _EPSILON

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
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

def executeProg(x_train,y_train,x_test,y_test,nb_epoch,noc,fold,bar,task):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=False,
        horizontal_flip=True,
        vertical_flip=True,
	    gamma_correction=False,
	    color_augment=False)
    datagen.fit(x_train)
    model = prepareResModel(); # Use prepareMobileModel() for Mobile Net
	
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(x_train)/ batch_size, epochs=nb_epoch,
                              validation_data=[x_test, y_test])
    history_Path = "./Result/{}_Fold{}_bar{}_Full_History.mat".format(task,fold+1,bar)
    sio.savemat(history_Path, mdict={'history': history.history});
    model_weight_Path = "./Result/{}_Fold{}_bar{}_Full_Weight.mat".format(task,fold+1,bar)
    model.save_weights(model_weight_Path)
    
    print("************************ Ended Full Finetuning of Leraning " + task+" Barrier "+ str(bar+1) + " Network of Fold "+str(fold+1)+" ************************");

#-----------------------Main Program---------------------------------

batch_size =4
nb_epoch = 100

DataPath='/home/cvpr/Anabik/SeverityExperimentMobileNet/Data/Psoriasis_Severity_Data_224.mat'
f = h5py.File(DataPath,'r');
totalImg = f.get('images')
label = f.get('labels')


Size=101;
for fold in range(7):
    testStartRange=Size*fold
    testEndRange=Size*(fold+1)
    testRange=range(testStartRange,testEndRange)
    trainRange=set(range(707))-set(testRange)
    trainRange=np.array(list(trainRange))
	
    #Train Data Preparation
    trainImg=totalImg[trainRange,:,:,:]
    trainImg=trainImg.astype('float32')
    trainLabel=label[trainRange,:]
    trainLabel=trainLabel.astype('int32')
    trainImg = np.swapaxes(trainImg, 1, 3)
	
    #Test Data Preparation
    testImg=totalImg[testRange,:,:,:]
    testImg=testImg.astype('float32')
    testLabel=label[testRange,:]
    testLabel=testLabel.astype('int32')    
    testImg = np.swapaxes(testImg, 1, 3)

    no_of_cls=5
    nb_epoch=[50,100,100,50]
    for bar in range(no_of_cls-1):
      trainEry=trainLabel[:,0]
      trainSca=trainLabel[:,1]      
      testEry=testLabel[:,0]
      testSca=testLabel[:,1]
         
      trainEry=np.float32(np.where(trainEry < bar+1, 0, 1))
      testEry=np.float32(np.where(testEry < bar+1, 0, 1))
      trainSca=np.float32(np.where(trainSca < bar+1, 0, 1))
      testSca=np.float32(np.where(testSca < bar+1, 0, 1))
      executeProg(trainImg,trainEry,testImg,testEry,nb_epoch[bar],5,fold,bar,"Ery")
      executeProg(trainImg,trainSca,testImg,testSca,nb_epoch[bar],5,fold,bar,"Sca")

    no_of_cls=4
    nb_epoch=[50,100,50]
    for bar in range(no_of_cls-1):
      trainInd=trainLabel[:,2]
      testInd=testLabel[:,2] 
      trainInd=np.float32(np.where(trainInd < bar+1, 0, 1))
      testInd=np.float32(np.where(testInd < bar+1, 0, 1))
      executeProg(trainImg,trainInd,testImg,testInd,nb_epoch[bar],4,fold,bar,"Ind")

