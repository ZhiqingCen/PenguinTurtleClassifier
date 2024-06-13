# penguins-and-turtle-classifier

A model that adds a bounding box and animal classification for the group project in COMP9517.

- Install dependencies: `pip install -r requirements.txt`

## Data Augmentation

- original dataset from [Penguins vs Turtles](https://www.kaggle.com/datasets/abbymorgan/penguins-vs-turtles)
- [preprocessing](preprocessing.ipynb)
  - retrieve training images from directory `/train` and `/train_annotations` from original dataset
  - usage: call function `generate_new_train_img(train_path)`, where `train_path` is the directory of train images
  - [augmented dataset can be found here](https://drive.google.com/file/d/13d7a9JlNWWpmOvTplTG-gbraqB6IqwM9/view?usp=sharing)
  - create a new directory `new` to store all augmented images and corresponding annotations file
    - `/new/penguin` contains original penguin images (excluding those images in `valid_penguin`) and augmentation images based on these original images
    - `/new/turtle` contains original penguin images (excluding those images in `valid_penguin`) and augmentation images based on these original images
    - `/new/valid_penguin` contains 25 randomly selected penguin images for validation purpose without any augmentation
    - `/new/valid_turtle` contains 25 randomly selected turtle images for validation purpose without any augmentation
- [mixup](mixup.ipynb)
  - [mixup dataset can be found here](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ETsACDwQOLxPqS7EHkhxcuoBHwh7lbQY1V-YSuxDMyz-vg?e=bdHCAD)
  - retrieve all original and augmented training images from directory `/new`
  - for all images, mix one penguin and one turtle image into one image using beta distribution, create `mixup_annotation` correspondingly
- [eda](eda.ipynb)
  - Exploratory Data Analysis using graphs and calculation on the original dataset
  - as well as showing augmented images

## Object Localisation

### YOLO

#### Dataset Requirement

- Dataset must be named as `datasets`
- Dataset must contain: `test`, `train`, `val`
- `datasets/train` must contain `penguin` and `turtle` folders
- `datasets` must contain `train_annotations`, `test_annotations`, and `val_annotations`
- When you run the notebook, `datasets/train_yolo` will be created from `datasets/train` folder with annotation files (bounding box coords and classification) and merging penguin and turtle images into the same folder.

#### Train & Inference

- To train, run the notebook
- You can change the pretrained model by modifying the class initilization parameter in cell 6
- To run inference with a different weight, reload the model in cell 8 with your trained weights. Trained weights and all plots are saved in runs/detect/train{int}
- The latest graph from training can be found in `YOLO_Training`

### DETR

- [best trained model can be found here](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/Ea4Ng46GVoBGtleqTGOKoZQB6KShSrmyQKnxLmCtChgBPw?e=B2CzA9)
- [object detection eval result can be found here](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/EbQdNmUaxdhFj5W4zFDr-8gBwLaiabpKF3gBA3hwTTV5VA?e=EYywS5)

- run [DETR.ipynb](DETR.ipynb)

  - `model_v2` is our [custom_detr](custom_detr.py) model, aiming to apply different iou (giou, diou & ciou) functions and mixup in the backend
  - [helper](helper.py) file is needed
  - input and output paths can be modified in cell 3, by default they are:

    ```python
    checkpoint = "facebook/detr-resnet-50"
    json_path = 'new/test_ann.json'
    train_path = 'new/train/'
    val_path = 'new/val/'
    test_path = 'new/test/'
    mixup_path = 'new/mixup/'
    output_dir = 'detr_ciou_finetuned_pvt2/'
    ```

  - training diary can be found in [DETR_Training_Notes.md](DETR_Training_Notes.md)
  - [TensorBoard](https://www.tensorflow.org/tensorboard) is used as visualisation tool
    - it can be accessed through [this link](http://127.0.0.1:6006/#timeseries) during training
  - after training model, two checkpoint will be saved to `output_dir`,
    - generally, one being the best checkpoint and one being the last checkpoint
    - note that if the last checkpoint is the best checkpoint being trained, then the previous checkpoint before the last checkpoint will also be saved
    - the best checkpoint can be determine by `Validation Loss` output during training
    - change the checkpoint in line `checkpoint_v2 = output_dir + 'checkpoint-5100'` in cell 10 from `checkpoint-5100` to the best checkpoint from the `output_dir`

- [crop train image](crop_train_image.py)

  - [cropped train images can be found here](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/Ec0vtRlXACNNsV_fx9ZGs-cBRIcx3lufzncdeTbwMK2ouw?e=N7Oerl)
  - crop trained image for later object classification step
  - input and output paths can be modified through

    ```python
    input_penguin_path = 'new/val/penguin/'
    input_turtle_path = 'new/val/turtle/'
    input_val_path='new/val/'
    input_test_path='new/test/'
    output_penguin_path = 'crop/test/penguin/'
    output_turtle_path = 'crop/test/turtle/'
    checkpoint_v2 = './detr_finetuned_pvt/checkpoint-2700'
    ```

  - where the `checkpoint-2700` should be change to the best checkpoint produced as mentioned above

## Object Clasification

- Dataset must be named as `data/crop`
- Dataset must contain: `test`, `train`, `val`
- `data/crop/train` must contain `penguin` and `turtle` folders
- At the completion of running the notebook, the model used will be saved and can be recovered using tensorflow
  - Model 1 `classifier_model_1.ipynb` will be saved as `classification_model_1` in the directory of the notebook
  - Model 2 `classifier_model_2.ipynb` will be saved as `demo_neural_net` in the directory of the notebook

## Demo

- To run the demo, please train DETR using `DETR.ipynb`, then update the variable `model_file` in cell 5 with your DETR weight path, then train the classification model and update the variable `classifier` in cell 6 with your classifier weight path.
