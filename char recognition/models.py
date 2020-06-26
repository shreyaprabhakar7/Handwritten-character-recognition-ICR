from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras import backend as K
import os

K.set_image_dim_ordering('th')

from keras.models import model_from_json

def get_model_config(model_name):
    model_config = {}
    if model_name == 'model1':
        model_config['model_builder'] = model1
    elif model_name == 'model2':
        model_config['model_builder'] = model2
    elif model_name == 'model3':
        model_config['model_builder'] = model3
    else:
        raise Exception('Unknown model name.')
    model_config['filepath_weight'] = os.path.join('../data/models', '{}_weight'.format(model_name))
    model_config['filepath_architechture'] = os.path.join('../data/models', '{}_model'.format(model_name))
    return model_config

def baseline_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape(int(image_shape[0] * image_shape[1]), input_shape = image_shape))
    model.add(Dense(128, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    return model

# def model1:
# def model2:

def save_model(model, model_config):
    #Saves model weights
    model.save_weights(model_config['filepath_weight'])
    print('Model weights saved in {}.'.format(model_config['filepath_weight']))

    #saves model architechture
    with open(model_config['filepath_architechture'], 'w') as outfile:
        outfile.write(model.to_json())
    print('Model architechture saved in {}.'.format(model_config['filepath_architechture']))

def load_model(filepath_weights, filepath_architechture):
    with open(filepath_architechture, 'r') as read:
        a = read.readlines()
        model = model_from_json(a[0])

    model.load_weights(filepath_weights, by_name=False)

    return model

def get_model_name():
    model_name = input('Select model:(baseline/simple_CNN/[model2])\n')
    if model_name == '':
        model_name = 'model2'
    return model_name
