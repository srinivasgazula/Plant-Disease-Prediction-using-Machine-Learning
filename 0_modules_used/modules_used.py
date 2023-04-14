#Mentiong all the modules that are used

# !pip3 install keras
# !pip3 install tensorflow

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Standardize the data
from sklearn.preprocessing import StandardScaler
# Modeling 
from sklearn.model_selection import train_test_split
# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, roc_curve, cohen_kappa_score, matthews_corrcoef, log_loss, make_scorer, recall_score, precision_score
from sklearn.metrics import SCORERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
# from io import StringIO
from sklearn import tree
import graphviz
# from IPython.display import Image 
from pydot import graph_from_dot_data
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

import pickle

import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.models import ResNet18_Weights
import warnings
import time
import os
import copy

from google.colab import drive
drive.mount('/content/drive')
# drive.mount('/content/drive', force_remount=True)

import warnings

import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, concatenate, Dropout,Concatenate
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import itertools

from tensorflow.keras.preprocessing.image import load_img, img_to_array
