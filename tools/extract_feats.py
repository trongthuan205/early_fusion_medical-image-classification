import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Dropout, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from keras.utils import plot_model

from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def extract_trans_feature(root_path, path_txt, model):
  annot = pd.read_csv(path_txt, header = None, sep=' ')
  imgs = np.array(annot.iloc[:,0].values)
  features = []
  for im in tqdm(imgs):
    img = Image.open(os.path.join(root_path, im)).convert('RGB')
    resize = transforms.Resize([224, 224])
    img = resize(img)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).to("cuda")
    tensor = tensor.unsqueeze(0).to("cuda")

    feature = model.extract_feat(tensor.to("cuda"), stage='neck')[0][0]
    for out in feature:
      out = out.cpu().detach().numpy()
      out = out.flatten()
      features.append(out)

  
  features = np.array(features)
  print(features.shape)
  return features

def extract_cnn_feature(root_path, path_txt, model):
  annot = pd.read_csv(path_txt, header = None, sep=' ')
  imgs = np.array(annot.iloc[:,0].values)
  features = []
  for im in tqdm(imgs):
    img = Image.open(os.path.join(root_path, im)).convert('RGB')
    resize = transforms.Resize([224, 224])
    img = resize(img)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).to("cuda")
    tensor = tensor.unsqueeze(0).to("cuda")

    feature = model.extract_feat(tensor.to("cuda"), stage='neck')
    for out in feature:
      out = out.cpu().detach().numpy()
      out = out.flatten()
      features.append(out)
  print(np.array(features).shape)
  return np.array(features)

def get_labels(path_txt):
  label = pd.read_csv(path_txt, header = None, sep=' ')
  label = np.array(label.iloc[:,-1].values)
  return label

def fuse_features(lst_feat):
  features = []
  for i in range(len(lst_feat[0])):
    feat_1 = lst_feat[0][i]
    feat_2 = lst_feat[1][i]
    fet = np.concatenate((feat_1, feat_2))
    features.append(fet)
  return np.array(features)

densenet201 = init_model( "ckpt_ccii/densenet201/densenet201_4xb256_in1k.py",
                        "ckpt_ccii/densenet201/best_accuracy_top-1_epoch_69.pth")
t2tvit = init_model("ckpt_ccii/t2t-vit-t-24_8xb64_in1k/t2t-vit-t-24_8xb64_in1k.py", 
		                "ckpt_ccii/t2t-vit-t-24_8xb64_in1k/best_accuracy_top-1_epoch_273.pth")

features_train_densenet201 = extract_cnn_feature("../data/Clean-CC-CCII/train", "../data/Clean-CC-CCII/train.txt", densenet201)
# np.save('train_fet_densenet201_ccii.npy', features_train_densenet201)
features_test_densenet201 = extract_cnn_feature("../data/Clean-CC-CCII/test", "../data/Clean-CC-CCII/test.txt", densenet201)
# np.save('test_fet_densenet201_ccii.npy', features_test_densenet201)
features_train_t2tvit = extract_trans_feature("../data/Clean-CC-CCII/train", "../data/Clean-CC-CCII/train.txt", t2tvit)
# np.save('train_fet_t2tvit_ccii.npy', features_train_t2tvit)
features_test_t2tvit = extract_trans_feature("../data/Clean-CC-CCII/test", "../data/Clean-CC-CCII/test.txt", t2tvit)
# np.save('test_fet_t2tvit_ccii.npy', features_test_t2tvit)

lst_feat_train = [features_train_densenet201, features_train_t2tvit]
lst_feat_test = [features_test_densenet201, features_test_t2tvit]
train_features = fuse_features(lst_feat_train)
np.save('train_fets_ccii.npy', features_train_t2tvit)
test_features = fuse_features(lst_feat_test)
np.save('test_fet_t2tvit_ccii.npy', features_train_t2tvit)

train_labels = get_labels("../data/Clean-CC-CCII/train.txt")
np.save('train_labels_ccii.npy',train_labels)
test_labels = get_labels("../data/Clean-CC-CCII/test.txt")
np.save('test_labels_ccii.npy',test_labels)








