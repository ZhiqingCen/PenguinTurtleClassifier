import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint, seed
from PIL import Image, ImageChops
import torch
from torchvision.transforms import ToPILImage, ColorJitter
from torchvision.transforms.functional import invert, posterize, solarize, hflip, vflip
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image

import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from detr import DetrImageProcessor, DetrForObjectDetection_v2
import json
import glob
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from PIL import Image, ImageDraw
from tqdm import tqdm
from datasets import load_dataset

from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers import Trainer, DetrConfig
from transformers import TrainingArguments

import albumentations
import evaluate

transform = transforms.Compose([
    transforms.PILToTensor()
])

# Get clean annotations
def clean_annotations():
    train_annotations = pd.read_json('datasets/train_annotations')

    train_annotations['category_id'] = train_annotations['category_id'].replace({1:0, 2:1})
    train_annotations['label'] = 'original'

    return train_annotations

# Data augmentation class
class GenerateImage:
    def __init__(self, img, image_id, category_id, bbox, area):
        self.img = img
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.height = 640 # height
        self.width = 640 # width
        self.area = area
        self.color_jitter = ColorJitter(saturation=.5, hue=.3) # satuation, contrast, brightness=.7
        self.path = ''
        self.which_path()
    
    def which_path(self):
        self.path = 'datasets/'

    def jitter_image(self):
        new_img = self.color_jitter(self.img)
        return [new_img, self.category_id, self.bbox, self.area, 'jitter']

    def inverted_image(self):
        new_img = invert(self.img)
        return [new_img, self.category_id, self.bbox, self.area, 'invert']

    def posterized_image(self):
        new_img = posterize(self.img, bits=3)
        return [new_img, self.category_id, self.bbox, self.area, 'posterize']

    def solarized_image(self):
        # not using this method
        new_img = solarize(self.img, threshold=210) # 192, 240
        return [new_img, self.category_id, self.bbox, self.area, 'solarize']

    def rotate_90_degree_image(self):
        new_img = self.img.rotate(90)
        # new_bbox = [int(self.bbox[1]), self.height-int(self.bbox[2]), int(self.bbox[3]), self.height-int(self.bbox[0])]
        new_bbox = [int(self.bbox[1]), self.width-(int(self.bbox[0])+int(self.bbox[2])), int(self.bbox[3]), int(self.bbox[2])]
        return [new_img, self.category_id, new_bbox, self.area, 'rotate90']
      
    def rotate_180_degree_image(self):
        new_img = self.img.rotate(180)
        # new_bbox = [self.width-int(self.bbox[2]), self.height-int(self.bbox[3]), self.width-int(self.bbox[0]), self.height-int(self.bbox[1])]
        new_bbox = [self.width-(int(self.bbox[0])+int(self.bbox[2])), self.height-(int(self.bbox[1])+int(self.bbox[3])), int(self.bbox[2]), int(self.bbox[3])]
        return [new_img, self.category_id, new_bbox, self.area, 'rotate180']
 
    def rotate_270_degree_image(self):
        new_img = self.img.rotate(270)
        # new_bbox = [self.width-int(self.bbox[3]), int(self.bbox[0]), self.width-int(self.bbox[1]), int(self.bbox[2])]
        new_bbox = [self.width-(int(self.bbox[1])+int(self.bbox[3])), int(self.bbox[0]), int(self.bbox[3]), int(self.bbox[2])]
        return [new_img, self.category_id, new_bbox, self.area, 'rotate270']

    def horizontally_flip_image(self):
        new_img = hflip(self.img)
        # new_bbox = [self.width-int(self.bbox[2]), int(self.bbox[1]), self.width-int(self.bbox[0]), int(self.bbox[3])]
        new_bbox = [self.width-(int(self.bbox[0])+int(self.bbox[2])), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])]
        return [new_img, self.category_id, new_bbox, self.area, 'hflip']

    def vertically_flip_image(self):
        new_img = vflip(self.img)
        # new_bbox = [int(self.bbox[0]), self.height-int(self.bbox[3]), int(self.bbox[2]), self.height-int(self.bbox[1])]
        new_bbox = [int(self.bbox[0]), self.height-(int(self.bbox[1])+int(self.bbox[3])), int(self.bbox[2]), int(self.bbox[3])]
        return [new_img, self.category_id, new_bbox, self.area, 'vflip']

    def jitter_vertically_flip_image(self):
        new_img = vflip(self.color_jitter(self.img))
        # new_bbox = [int(self.bbox[0]), self.height-int(self.bbox[3]), int(self.bbox[2]), self.height-int(self.bbox[1])]
        new_bbox = [int(self.bbox[0]), self.height-(int(self.bbox[1])+int(self.bbox[3])), int(self.bbox[2]), int(self.bbox[3])]
        return [new_img, self.category_id, new_bbox, self.area, 'jitter_vflip']

    def jitter_180_degree_image(self):
        new_img = self.color_jitter(self.img).rotate(180)
        # new_bbox = [self.width-int(self.bbox[2]), self.height-int(self.bbox[3]), self.width-int(self.bbox[0]), self.height-int(self.bbox[1])]
        new_bbox = [self.width-(int(self.bbox[0])+int(self.bbox[2])), self.height-(int(self.bbox[1])+int(self.bbox[3])), int(self.bbox[2]), int(self.bbox[3])]
        return [new_img, self.category_id, new_bbox, self.area, 'jitter180']
    
# Apply data augmentation to given image
def preprocessing_demo(image, image_id):
    ret = []

    train_annotations = clean_annotations()

    category_id = train_annotations.iloc[image_id]['category_id']
    area = train_annotations.iloc[image_id]['area']
    bbox = train_annotations.iloc[image_id]['bbox']

    generator = GenerateImage(image, image_id, category_id, bbox, area)
    
    transformed = [
        generator.jitter_image(),
        generator.inverted_image(),
        generator.posterized_image(),
        generator.rotate_90_degree_image(),
        generator.rotate_180_degree_image(),
        generator.rotate_270_degree_image(),
        generator.horizontally_flip_image(),
        generator.jitter_vertically_flip_image(),
        generator.jitter_180_degree_image()
    ]

    return transformed

# Transform the image with augmentation and plot
def transform_image(file_name, index):
    transformed = preprocessing_demo(Image.open(file_name), index)

    fig, axes = plt.subplots(nrows=3, ncols= 3, figsize=(12, 12))
    ax = axes.ravel()

    for i,transform_name in enumerate(['jitter', 'inverted', 'posterized', 'rotate 90', 'rotate 180', 'rotate 270', 'horizontally flip', 'jitter vertically flip', 'jittter 180 degree']):

        box = box_convert(torch.tensor(transformed[i][2]), 'xywh', 'xyxy')
        box = box[None, :]

        img = draw_bounding_boxes(transform(transformed[i][0]), box, width=5, colors='blue')
        img = np.moveaxis(np.array(img), 0, -1)

        ax[i].set_title(transform_name)

        ax[i].imshow(img)
        ax[i].axis('off')

    plt.suptitle('Data Augmentation Techniques')
    plt.show()
    
    return transformed

# Apply bounding box regression on one testing file
def inference(model, device, image_processor, image, ground_truth_bbox=None, draw=True):
    '''Only for files in testing folder'''
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        pixel_mask = inputs["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    
    if draw:
        draw = ImageDraw.Draw(image)
        g_x, g_y, g_x2, g_y2 = tuple(ground_truth_bbox)
        draw.rectangle((g_x, g_y, g_x + g_x2, g_y + g_y2), outline="blue", width=2)

        score, label, box = max(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0])
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)

        plt.imshow(image)
        plt.show()
    
    return model.config.id2label[label.item()], round(score.item(), 3), box

# Apply bounding box regression on 9 testing files
def inference_multiple(model, device, image_processor, images, ground_truth_bboxes):
    '''Only for files in testing folder'''

    fig, axes = plt.subplots(nrows=3, ncols= 3, figsize=(12, 12))
    ax = axes.ravel()
    
    original_images = [img.copy() for img in images]
    
    bboxes = []
    
    for i in range(len(images)):
        image = images[i]
        ground_truth_bbox = ground_truth_bboxes[i]
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            pixel_mask = inputs["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

        draw = ImageDraw.Draw(image)
        g_x, g_y, g_x2, g_y2 = tuple(ground_truth_bbox)
        draw.rectangle((g_x, g_y, g_x + g_x2, g_y + g_y2), outline="blue", width=2)

        score, label, box = max(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0])
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        bboxes.append(box)
        
        draw.rectangle((x, y, x2, y2), outline="red", width=1)

        ax[i].imshow(image)
        ax[i].axis('off')

    plt.show()
    
    return original_images, bboxes

# Crop image with bounding boxes
def crop_from_bboxes(images, bboxes):
    new_images = []
    for i in range(len(images)):
        new_images.append(images[i].crop(bboxes[i]))
        
    fig, axes = plt.subplots(nrows=3, ncols= 3, figsize=(12, 12))
    ax = axes.ravel()

    for i,transform_name in enumerate(['jitter', 'inverted', 'posterized', 'rotate 90', 'rotate 180', 'rotate 270', 'horizontally flip', 'jitter vertically flip', 'jittter 180 degree']):
        ax[i].set_title(transform_name)

        ax[i].imshow(new_images[i])
        ax[i].axis('off')

    plt.suptitle('Croped images')
    plt.show()
    
    return new_images

# Perform prediction for a list of 9 images
def predict_from_list(classifier, images):
    classified = []

    fig, axes = plt.subplots(nrows=3, ncols= 3, figsize=(12, 12))
    ax = axes.ravel()

    for i in range(9):
        animal = classify_image(classifier, images[i])
        classified.append(animal)
        ax[i].set_title(animal)

        ax[i].imshow(images[i])
        ax[i].axis('off')

    plt.suptitle('Croped images')
    plt.show()
    
    return classified

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Classify images
def classify_image(classifier, img):
    image = img_to_array(tf.image.resize(img, (96, 96)))
    image_array = np.array([image])

    return "Penguin" if classifier.predict(image_array) < 0 else "Turtle"