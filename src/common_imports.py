# Built in packages
import os
import datetime
from enum import Enum

# Third party packages
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

USING_TENSORFLOW_2_1 = True
FORCE_TENSORFLOW_CPU = False
TENSORFLOW_LOGGING = 1  # 0 - disabled, 1 - default, 2 - verbose

if USING_TENSORFLOW_2_1:
    os.environ['CUDA_PATH'] = os.environ['CUDA_PATH_V10_1']
else:
    os.environ['CUDA_PATH'] = os.environ['CUDA_PATH_V10_0']
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CUDA_PATH']

if TENSORFLOW_LOGGING == 0:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
elif TENSORFLOW_LOGGING == 1:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
elif TENSORFLOW_LOGGING == 2:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

#os.environ["TF_CPP_VMODULE"] = 'gpu_device=5'

if FORCE_TENSORFLOW_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import tensorflow as tf
from tensorflow.keras import optimizers, metrics, layers
import tensorflow_addons as tfa

for device in tf.config.experimental.list_physical_devices('GPU'):
    print('TF GPU device: ' + str(device))
    #tf.config.experimental.set_memory_growth(device, True)

print('\n\n')
print('CUDA_PATH=%s' % (os.environ['CUDA_PATH']))
print('Tensorflow Version: %s ' % (tf.__version__))
print('Tensorflow_addons Version: %s' % (tfa.__version__))
print('\n\n')

import spacy

spacy.prefer_gpu()
NLP = spacy.load("en_core_web_lg")

