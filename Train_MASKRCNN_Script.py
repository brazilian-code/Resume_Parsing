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
        
        dir_list = os.listdir(images_dir)
        count = 0
        image_id = ""

        for filename in dir_list:
            # Image ID is file name without .jpg
            image_id = filename[:-4]

            if is_train and count <= len(dir_list)*0.75:
                continue
        
            if not is_train and count > len(dir_list)*0.75:
                continue
            count+=1

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

    STEPS_PER_EPOCH =  131 
    


def train_model(dataset_path, model_path, num_epochs, final_model_path):

    # Training
    train_dataset = ResumeDataset()
    train_dataset.load_dataset(dataset_dir=dataset_path, is_train=True)
    train_dataset.prepare()
    # Validation
    validation_dataset = ResumeDataset()
    validation_dataset.load_dataset(dataset_dir=dataset_path, is_train=False)
    validation_dataset.prepare()
    
    #For Training;
    config = ResumeConfig()
    
    model = mrcnn.model.MaskRCNN(mode='training', 
                                 model_dir='.log', 
                                 config=config)
    model.keras_model.summary()
    
    model.load_weights(filepath=model_path, by_name=True)
    
    print("Weights loaded!")
    
    print("Training Started!")
    model.train(train_dataset=train_dataset, 
                val_dataset=validation_dataset, 
                learning_rate=config.LEARNING_RATE, 
                epochs=num_epochs,
                layers='heads')
    

    model.keras_model.save_weights(final_model_path)
    print("Model Saved!")
