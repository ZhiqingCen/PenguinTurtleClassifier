import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

input_penguin_path = 'new/val/penguin/'
input_turtle_path = 'new/val/turtle/'
input_val_path='new/val/'
input_test_path='new/test/'
output_penguin_path = 'crop/test/penguin/'
output_turtle_path = 'crop/test/turtle/'

val_annotations = pd.read_json('new/valid_annotations')
test_annotations = pd.read_json('new/test_annotations')
train_annotations = pd.read_json('new/train_annotations')

device = 'mps'
checkpoint_v2 = './detr_finetuned_pvt/checkpoint-2700'
image_processor = DetrImageProcessor.from_pretrained(checkpoint_v2)
model = DetrForObjectDetection.from_pretrained(checkpoint_v2)
model.to(device)
images = os.listdir(input_test_path)

# Crop all the images according to bounding box

for img_name in tqdm(images):
    if img_name.startswith('.'):
        continue
    num = int(img_name.split('.')[0].split('_')[-1])
    image_df = test_annotations[test_annotations['image_id'] == num].values[0]
    category_id = image_df[2]
    with Image.open(input_test_path + img_name) as image:
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            pixel_mask = inputs["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0, target_sizes=target_sizes)[
                0]
        best_zip = sorted(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0])
        score, label, box = max(zip(results["scores"], results["labels"], results["boxes"]), key=lambda x: x[0])
        box = [round(i, 2) for i in box.tolist()]

        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at image box{box}"
        )
        x, y, x2, y2 = tuple(box)

        image = image.crop((x, y, x2, y2))
        if category_id == 1:
            image.save(output_penguin_path + img_name)
        if category_id == 2:
            image.save(output_turtle_path + img_name)
