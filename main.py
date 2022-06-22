from data_reading import gather_paths_all,gather_images_from_paths
from finetuning import finetune,pred_mid


model_names=["mobilenetv2"]#,"mobilenetv2","densenet169","vgg19","inceptionv3","resnet50","resnet152","nasnetlarge"]
data_names=["me2018","ed2020"]#"me2017",
model_name="resnet50"
data_name="me2018"

import numpy as np
import os
import cv2, numpy as np, os, h5py, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

data_path_test_2017="/kaggle/input/kvasirv1-val/val/"
data_path_train_2017="/kaggle/input/kvasirv1train/dev/"
data_path_test_2018="/kaggle/input/kvasirv2val/val/"
data_path_train_2018="/kaggle/input/kvasirv2-dev/dev/"
data_path_test_2020="/kaggle/input/hyperkvasir-val/val/"
data_path_train_2020="/kaggle/input/hyperkvasir-dev/dev/"

batch_size = 25
nb_epoch = 500
learning_rate=0.01
#optimiser="adagard"
optimiser="sgd"


label_map36=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3','cecum', 'normal-cecum', 'dyed-lifted-polyps', 'dyed-resection-margins',
         'esophagitis','esophagitis-a','esophagitis-b-d','hemorrhoids', 'ileum', 'impacted-stool','normal-z-line','polyps','pylorus','normal-pylorus',
         'retroflex-rectum','retroflex-stomach','ulcerative-colitis','ulcerative-colitis-0-1','ulcerative-colitis-1-2','ulcerative-colitis-2-3',
         'ulcerative-colitis-grade-1','ulcerative-colitis-grade-2','ulcerative-colitis-grade-3',
         'lesion', 'dysplasia', 'cancer', 'blurry-nothing', 'colon-clear', 'stool-inclusions', 'stool-plenty', 'instruments', 'out-of-patient']

label_map23=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
             'hemorrhoids', 'ileum', 'impacted-stool', 'normal-z-line', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach',
             'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-3'] 
label_map16=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
label_map8=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']

from keras.layers import Input, merge, ZeroPadding2D, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import Sequential, backend as K, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, Adagrad
from keras.applications.densenet import DenseNet169
from keras.applications.resnet import ResNet152, ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications import NASNetLarge, MobileNetV2, InceptionV3
for model_name in model_names:
    for data_name in data_names:
        if(data_name=="me2017"):
            label_map=label_map8
            train_count=4000
            test_count=4000
            num_classes=8
            dataset="media_eval_2017"
            data_path_train=data_path_train_2017
            data_path_test=data_path_test_2017
        if(data_name=="me2018"):
            label_map=label_map16
            train_count=5293
            test_count=8740
            num_classes=16
            dataset="media_eval_2018"
            data_path_train=data_path_train_2018
            data_path_test=data_path_test_2018
        if(data_name=="ed2020"):
            label_map=label_map23
            train_count=10662
            test_count=721
            num_classes=23
            dataset="endo_tech_2020"
            data_path_train=data_path_train_2020
            data_path_test=data_path_test_2020
        last_layer='avg_pool'
        if(model_name=="densenet169"):
            base_model=DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='avg_pool'
        if(model_name=="nasnetlarge"):
            base_model=NASNetLarge(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='global_average_pooling2d' 
        if(model_name=="mobilenetv2"):
            base_model=MobileNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='global_average_pooling2d' 
        if(model_name=="vgg19"):
            base_model=VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='fc1'
        if(model_name=="resnet50"):
            base_model=ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='avg_pool'
        if(model_name=="resnet152"):
            base_model=ResNet152(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='avg_pool'
        if(model_name=="inceptionv3"):
            base_model=InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            last_layer='avg_pool'

        weights_path_fine="/kaggle/working/"+data_name+"_"+model_name+"_"+str(nb_epoch)+"_updatingLR.h5"
        model_updated=finetune(base_model=base_model,output_layer=last_layer,weights='imagenet',weights_new=weights_path_fine,nb_epoch=nb_epoch,data_path=data_path_train,num_classes=num_classes)
        
        
        
        paths,Y,Ys=gather_paths_all(data_path_train,num_classes=num_classes)
        X=gather_images_from_paths(paths,0,len(paths),img_rows=224,img_cols=224)
        
        img_labels=[p.split("/")[-2]+"__"+p.split("/")[-1] for p in paths]
        
        Y_preds=pred_mid(model_updated,-2,X)
        #Y_preds=model_updated.predict(X)
        
        
        df=pd.DataFrame(Y_preds)
        df["image_name"]=img_labels
        df["Actual"]=Y
        df["Pred"]=[np.argmax(Y_preds[i,:])+1 for i in range(len(Y_preds))]
        df.to_csv("/kaggle/working/"+data_name+"_"+model_name+"_"+str(nb_epoch)+"_updatingLR_train.csv")
        print(accuracy_score(df["Actual"],df["Pred"]))
        
        paths,Y,Ys=gather_paths_all(data_path_test,num_classes=num_classes)
        X=gather_images_from_paths(paths,0,len(paths),img_rows=224,img_cols=224)
        
        img_labels=[p.split("/")[-2]+"__"+p.split("/")[-1] for p in paths]
        
        #Y_preds=model_updated.predict(X)
        Y_preds=pred_mid(model_updated,-2,X)
        
        df=pd.DataFrame(Y_preds)
        df["image_name"]=img_labels
        df["Actual"]=Y
        df["Pred"]=[np.argmax(Y_preds[i,:])+1 for i in range(len(Y_preds))]
        df.to_csv("/kaggle/working/"+data_name+"_"+model_name+"_"+str(nb_epoch)+"_updatingLR_test.csv")
        print(accuracy_score(df["Actual"],df["Pred"]))
        
    
#model_updated=finetune_h5(base_model=base_model,output_layer='avg_pool',weights='imagenet',include_top=True,weights_new=weight_path_ft,nb_epoch=nb_epoch,data_file=path_train,chunk_size=step_size)