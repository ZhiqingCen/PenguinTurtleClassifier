import json
import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
import torchvision
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss

from transformers import DetrForObjectDetection, DetrImageProcessor, DetrModel, DetrPreTrainedModel
from transformers import Trainer, DetrConfig
from transformers import TrainingArguments
from transformers.image_transforms import center_to_corners_format
from transformers.utils import ModelOutput

from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from tqdm import tqdm
import albumentations
import evaluate
from datasets import load_dataset

from dataclasses import dataclass
from ultralytics import YOLO

# Convert dataset and annotation to COCO JSON format
def transform_aug_ann(example, annotations, transform, image_processor):
    image_id = int(example["image"][0].filename.split(".")[0].split("_")[-1])
    # images, bboxes, area, categories = [], [], [], []
    image_df = annotations[annotations["image_id"] == image_id].values[0]
    bbox = image_df[2]
    category_id = image_df[1]
    area = image_df[3]
    image = np.array(example["image"][0].convert("RGB"))[:, :, ::-1]
    try:
        transform(image=image, bboxes=[bbox], category=[category_id])
    except Exception as e:
        print(e)
        print(image_df)

    target = [
        {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "isCrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
            ],
        }
    ]
    return image_processor(images=[image], annotations=target, return_tensors="pt")

# Convert dataset and annotation to COCO JSON format for mixup
def transform_mixup_ann(example, annotations, transform, image_processor):
    file_name = example["image"][0].filename
    image_id = int(file_name.split(".")[0].split("_")[-1])
    image_df = annotations[annotations["image_id"] == image_id].values[0]
    bbox = image_df[2]
    category_id = image_df[1]
    area = image_df[3]
    image = np.array(example["image"][0].convert("RGB"))[:, :, ::-1]
    try:
        transform(image=image, bboxes=bbox, category=category_id)
    except Exception as e:
        print(e)
        print(image_df)

    annotations = []
    for i in range(0, len(category_id)):
        new_ann = {
            "image_id": image_id,
            "category_id": 2,
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)
    target = [{"image_id": image_id, "annotations": annotations}]
    return image_processor(images=[image], annotations=target, return_tensors="pt")

# DETR compute validation loss
def compute_metrics(eval_pred):
    outputs, labels = eval_pred
    result_dict = {}
    for key, value_array in outputs[0].items():
        mean_value = np.mean(value_array)
        result_dict[key] = mean_value
    return result_dict

# DETR Batch processing
def collate_fn_onehot(batch, image_processor, mixup_annotation):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = []
    for item in batch:
        l = dict(item["labels"])
        if len(l["class_labels"]) == 1:
            l["class_labels"] = F.one_hot(l["class_labels"], num_classes=3)
        else:
            image_df = mixup_annotation[
                mixup_annotation["image_id"] == int(l["image_id"][0])
            ].values[0]
            class_labels = list(image_df[1])
            class_labels.append(0)
            l["class_labels"] = torch.tensor([class_labels], dtype=torch.float)
        labels.append(l)
    batch = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }
    return batch

# DETR Batch processing
def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [dict(item["labels"]) for item in batch]
    batch = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }
    return batch

# Detection performance: central distance L2
def detect_perform(pred_y, true_y):
    """
    pred_y, true_y: array-like (n_samples, n_features=4)
    """
    pred_x_y_centre = np.array(
        [pred_y[:, 0] + pred_y[:, 2] / 2, pred_y[:, 1] + pred_y[:, 3] / 2]
    )
    true_x_y_centre = np.array(
        [true_y[:, 0] + true_y[:, 2] / 2, true_y[:, 1] + true_y[:, 3] / 2]
    )
    perform = np.linalg.norm(pred_x_y_centre - true_x_y_centre, axis=0)

    return perform, np.mean(perform), np.std(perform)

# Intersection of two images
def intersection(pred_y, true_y):
    """
    pred_y, true_y: array-like (n_samples, n_features=4)
    """
    pred_x_min, pred_y_min, pred_width, pred_height = (
        pred_y[:, 0],
        pred_y[:, 1],
        pred_y[:, 2],
        pred_y[:, 3],
    )
    true_x_min, true_y_min, true_width, true_height = (
        true_y[:, 0],
        true_y[:, 1],
        true_y[:, 2],
        true_y[:, 3],
    )

    pred_x_max, pred_y_max = pred_x_min + pred_width, pred_y_min + pred_height
    true_x_max, true_y_max = true_x_min + true_width, true_y_min + true_height

    # int for intersection
    int_x_y_min = np.maximum([pred_x_min, pred_y_min], [true_x_min, true_y_min])
    int_x_y_max = np.minimum([pred_x_max, pred_y_max], [true_x_max, true_y_max])

    # intersection area
    return (int_x_y_max[0] - int_x_y_min[0]) * (int_x_y_max[1] - int_x_y_min[1])

# Compute IOU
def iou(pred_y, true_y, pred_area, true_area):
    """
    pred_y, true_y: array-like (n_samples, n_features=4)
    pred_area, true_area: array-like (n_samples)
    """
    intersect = intersection(pred_y, true_y)
    union = pred_area + true_area - intersect
    perform = intersect / union
    return perform, np.mean(perform), np.std(perform)

# Get testing file name
def to_file_name(id_, test_path):
    id_ = str(id_)
    return test_path + f"/image_id_{'0' * (3 - len(id_)) + id_}.jpg"

# Perform inference using DETR on a test image
def inference(model, device, image_processor, image_path, test_annotations):
    '''Only for files in testing folder'''
    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        pixel_mask = inputs["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )[0]

    draw = ImageDraw.Draw(image)
    image_id = int(image.filename.split(".")[0].split("_")[-1])
    ground_truth_bbox = test_annotations[
        test_annotations["image_id"] == image_id
    ].values[0][3]
    g_x, g_y, g_x2, g_y2 = tuple(ground_truth_bbox)
    draw.rectangle((g_x, g_y, g_x + g_x2, g_y + g_y2), outline="blue", width=2)

    # best_zip = sorted(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0])
    score, label, box = max(
        zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0]
    )
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

# Get performance of DETR using test dataset
def get_performance(model, device, image_processor, test_annotations, test_path):
    pred_y_arr = []
    pred_area_arr = []

    true_y_arr = np.array(test_annotations["bbox"].tolist())
    true_area_arr = test_annotations["area"].to_numpy()

    for id_ in tqdm(test_annotations.image_id.to_list()):
        image = Image.open(to_file_name(id_, test_path))

        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            pixel_mask = inputs["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([image.size[::-1]])
            pred = image_processor.post_process_object_detection(
                outputs, threshold=0, target_sizes=target_sizes
            )[0]

        score, label, box = max(
            zip(pred["scores"], pred["labels"], pred["boxes"]), key=lambda x: x[0]
        )

        pred_y = box.cpu().numpy()
        pred_y[2] = pred_y[2] - pred_y[0]
        pred_y[3] = pred_y[3] - pred_y[1]

        pred_area = pred_y[2] * pred_y[3]

        pred_y_arr.append(pred_y)
        pred_area_arr.append(pred_area)

    pred_y_arr = np.array(pred_y_arr)
    pred_area_arr = np.array(pred_area_arr)

    detection_arr, detect_perm_mu, detect_perm_std = detect_perform(
        pred_y_arr, true_y_arr
    )
    iou_arr, iou_mu, iou_std = iou(pred_y_arr, true_y_arr, pred_area_arr, true_area_arr)

    print(f"Detection Performance mu: {detect_perm_mu:.3f}, std: {detect_perm_std:.3f}")
    print(f"IOU mu: {iou_mu:.3f}, std: {iou_std:.3f}")

    fig, axs = plt.subplots(nrows=2)

    axs[0].hist(detection_arr, bins=10)
    axs[0].set_title('Detection Performance Histogram')
    axs[1].hist(iou_arr, bins=10)
    axs[1].set_title('IOU')
    plt.tight_layout()
    plt.show()

# Convert category with 0 penguin 1 turtle
def update_category(val):
    return val -1 

# Get ID from file name
def get_id(file_name):
    text = file_name.split('/')[2]
    id_ = text.split('_')[2]
    return int(id_.split('.')[0])

# Generate YOLO annotation given input image folder and annotation
def gen_ann_yolo_txt(input_folder, output_folder, ann):
    for file_name in glob.glob(input_folder + '/image_id_*.jpg'):
        id_ = get_id(file_name)

        img = cv.imread(file_name)
                
        bbox = ann[ann['image_id'] == id_]['bbox'].tolist()[0]
        cat_id = ann[ann['image_id'] == id_]['category_id'].tolist()[0]
        
        txt_file = file_name.split('.')[0] + '.txt'
        
        img_width = img.shape[0]
        img_height = img.shape[1]
        
        # https://stackoverflow.com/questions/64238660/convert-a-csv-file-to-yolo-darknet-format
        # https://github.com/AlexeyAB/Yolo_mark/issues/60
        # https://docs.ultralytics.com/datasets/detect/#:~:text=Ultralytics%20YOLO%20format&text=txt%20file%20is%20required.,(from%200%20%2D%201).
        yolo_bbox = [(bbox[0] + bbox[2]/2)/img_width, (bbox[1] + bbox[3]/2)/img_height, bbox[2]/img_width, bbox[3]/img_height]
        
        # class x_center y_center width height        
        with open(txt_file, "w") as f:
            f.write(f"{cat_id} {' '.join([str(i) for i in yolo_bbox])}")

# Freeze YOLOv8n backbone layers
def freeze_layer(trainer):
    # https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/#before-you-start
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)] # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

# Get YOLO performance
def get_performance_yolo(model, test_annotations, test_path):
    pred_y_arr = []
    pred_area_arr = []

    true_y_arr = np.array(test_annotations['bbox'].tolist())
    true_area_arr = test_annotations['area'].to_numpy()

    for id_ in test_annotations.image_id.to_list():
        file = to_file_name(id_, test_path)
        results = model.predict(file)
        
        try:
            pred_y = results[0].boxes[0].xyxy[0].tolist()
        except:
            pred_y = [160,160,640,640]

        pred_y[2] = pred_y[2] - pred_y[0]
        pred_y[3] = pred_y[3] - pred_y[1]
            
        pred_area = pred_y[2] * pred_y[3]

        pred_y_arr.append(pred_y)
        pred_area_arr.append(pred_area)

    pred_y_arr = np.array(pred_y_arr)
    pred_area_arr = np.array(pred_area_arr)

    detection_arr, detect_perm_mu, detect_perm_std = detect_perform(pred_y_arr, true_y_arr)
    iou_arr, iou_mu, iou_std = iou(pred_y_arr, true_y_arr, pred_area_arr, true_area_arr)

    print(f'Detection Performance mu: {detect_perm_mu}, std: {detect_perm_std}')
    print(f'IOU mu: {iou_mu}, std: {iou_std}')

    fig, axs = plt.subplots(nrows=2)

    axs[0].hist(detection_arr, bins=10)
    axs[0].set_title('Detection Performance Histogram')
    axs[1].hist(iou_arr, bins=10)
    axs[1].set_title('IOU')
    plt.tight_layout()
    plt.show()

# DETR save evaluation image
def save_eval_image(test_path, image_processor, device, test_annotations, model):
    images = os.listdir(test_path)
    for img_name in tqdm(images):
        if img_name.startswith("."):
            continue
        image = Image.open(test_path + img_name)
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            pixel_mask = inputs["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.1, target_sizes=target_sizes
            )[0]
        draw = ImageDraw.Draw(image)
        image_id = int(image.filename.split(".")[0].split("_")[-1])
        ground_truth_bbox = test_annotations[
            test_annotations["image_id"] == image_id
        ].values[0][3]
        g_x, g_y, g_x2, g_y2 = tuple(ground_truth_bbox)
        draw.rectangle((g_x, g_y, g_x + g_x2, g_y + g_y2), outline="blue", width=2)

        score, label, box = max(
            zip(results["scores"], results["labels"], results["boxes"]),
            key=lambda x: x[0],
        )
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        image.save("eval/" + img_name)