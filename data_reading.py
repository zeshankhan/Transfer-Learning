# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:38:56 2022

@author: ZESHAN KAHN
"""

import cv2, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
def gather_paths_all(jpg_path,num_classes=16):
  i=0
  if(num_classes==16):
    label_map=label_map16
  elif(num_classes==8):
    label_map=label_map8
  elif(num_classes==36):
    label_map=label_map36
  elif(num_classes==23):
    label_map=label_map23
  
  
  folder=os.listdir(jpg_path)
  count=0
  if (os.path.isfile(jpg_path+folder[0])):
    count=len(os.listdir(jpg_path))
  else:
    count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
  ima=['' for x in range(count)]
  labels=np.zeros((count,num_classes),dtype=float)
  label=[0 for x in range(count)]
  if (os.path.isfile(jpg_path+folder[0])):
    for f in folder:
      im=jpg_path+f
      ima[i]=im
      label[i]=0
      i+=1
      if(count<i):
        break
  else:
    for fldr in folder:
      for f in os.listdir(jpg_path+fldr+"/"):
          im=jpg_path+fldr+"/"+f
          ima[i]=im
          label[i]=label_map.index(fldr)+1
          i+=1
      if(count<=i):
          break
  for i in range(count):
      labels[i][label[i]-1]=1
  return ima,label,labels

def gather_images_from_paths(jpg_path,start,count,img_rows=224,img_cols=224):
    if(model_name=="nasnetlarge"):
        img_rows=img_cols=331
    print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
    ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
    for i in range(count):
        img=cv2.imread(jpg_path[start+i])
        im = cv2.resize(img, (img_rows, img_cols)).astype(np.uint8)
        ima[i]=im
    return ima