import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import src.utils import utils
import src.config as cfg
from src.yolov3 import YOLOv3, decode


INPUT_SIZE = 416
CHANNEL_SIZE = 3
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

predicted_dir_path = "../mAP/predicted"
ground_truth_dir_path = "../mAP/ground-truth"
if os.path.exists(predicted_dir_path):
    shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path):
    shutil.rmtree(ground_truth_dir_path)
if os.path.exists(cfg.TEST.DETECTED_IMAGE_PATH):
    shutil.rmtree(cfg.TEST.DETECTED_IMAGE_PATH)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(cfg.TEST.DETECTED_IMAGE_PATH)

# Build the model
input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, CHANNEL_SIZE])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./yolov3")

with open(cfg.TEST.ANNOT_PATH, "r") as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        file_name = image_path.split("/")[-1]
        np_img = cv2.imread(image_path)
        np_rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(","))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_path, str(num) + ".txt")
