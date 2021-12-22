# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 02:28:40 2021

@author: davba
"""
#Notebook written by David A. A. Balaban
from pdf2image import convert_from_path 
import easyocr
import numpy as np
import PIL # Python Imaging Library
from PIL import ImageDraw # drawing bounding boxes
import tensorflow as tf
from IPython.display import display,Image
from matplotlib.pyplot import imshow
import xml.dom.minidom
import pandas as pd
import mrcnn
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import urllib.request
import os
import xml.etree


#Using Keras==2.2.5

# Sections = Personal Info, Education, Skills, Projects, Work Experience, Extra

class ResumeDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adding all possible sections
        self.add_class("dataset", 1, "Personal Info")
        self.add_class("dataset", 2, "Education")
        self.add_class("dataset", 3, "Skills")
        self.add_class("dataset", 4, "Projects")
        self.add_class("dataset", 5, "Work Experience")
        self.add_class("dataset", 6, "Extra")
        
        
        
        images_dir = dataset_dir + '\\Resumes\\'
        annotations_dir = dataset_dir + '\\Resume_Annotations\\'

        for filename in os.listdir(images_dir):
            # Image ID is file name without .jpg
            image_id = filename[:-4]
            #Lets leave resume s107 for validation
            if is_train and int(image_id[1:]) >= 85:
                continue

            if not is_train and int(image_id[1:]) < 85:
                continue


            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            obj = boxes[i]
            box = obj[1]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(obj[0]))
        return masks, np.asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for obj in root.findall('./object'):
        	name = obj.find('name').text
        	xmin = int(obj.find('bndbox/xmin').text)
        	ymin = int(obj.find('bndbox/ymin').text)
        	xmax = int(obj.find('bndbox/xmax').text)
        	ymax = int(obj.find('bndbox/ymax').text)
        	coors = [xmin, ymin, xmax, ymax]
        	box_array = [name,coors]
        	print(box_array)
        	boxes.append(box_array)    

            
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class ResumeConfig(mrcnn.config.Config):
    NAME = "resumes_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 7
    
    LEARNING_RATE = 0.001

    STEPS_PER_EPOCH =  10
    


def train_model(model_path, num_epochs, final_model_path):

    # Training
    train_dataset = ResumeDataset()
    train_dataset.load_dataset(dataset_dir=r'D:\\ResumeIT', is_train=True)
    train_dataset.prepare()
    # Validation
    validation_dataset = ResumeDataset()
    validation_dataset.load_dataset(dataset_dir=r'D:\\ResumeIT', is_train=False)
    validation_dataset.prepare()
    
    #For Training;
    config = ResumeConfig()
    
    model = mrcnn.model.MaskRCNN(mode='training', 
                                 model_dir='.log', 
                                 config=config)
    model.keras_model.summary()
    
    #r'D:\\ResumeIT\\RESUMEIT_Model_20Epochs_262Steps.h5'
    model.load_weights(filepath=model_path, by_name=True)
    
    print("Weights loaded!")
    
    print("Training Started!")
    model.train(train_dataset=train_dataset, 
                val_dataset=validation_dataset, 
                learning_rate=config.LEARNING_RATE, 
                epochs=num_epochs,
                layers='heads')
    
    model_path = r'D:\\ResumeIT\\RESUMEIT_Model_Finalized.h5'
    model.keras_model.save_weights(final_model_path)
    print("Model Saved!")

#train_model("D:\\ResumeIT\\RESUMEIT_Model_Finalized.h5", 1, "D:\\ResumeIT\\testing.h5")

'''
#For Predicting

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

CLASS_NAMES = ['BG', 'Personal Info', 'Education', 'Skills', 'Projects', 'Work Experience', 'Extra']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)

model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath= r'D:\\ResumeIT\\RESUMEIT_Model_11_30_20Epochs_131Steps.h5', 
                   by_name=True)

image = cv2.imread(r'D:\\ResumeIT\\old_Resumes_11_30\\s109.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)

r = r[0]

mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])

'''