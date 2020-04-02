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

        print("=> ground truth of %s:" % file_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, "w") as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = " ".join([class_name, xmin, ymin, xmax, ymax]) + "\n"
                f.write(bbox_mess)
                print("\t" + str(bbox_mess).strip())

        print("=> predict result of %s:" % file_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + ".txt")
        # need to check the data dims
        image_size = np_rgb_img.shape[:2]
        image_data = utils.image_preprocess(np.copy(np_rgb_img), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # need to investigate the following part ---from here
        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        # ---ends here
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method="nms")

        if cfg.TEST.DETECTED_IMAGE_PATH is not None:
            image = utils.draw_bbox(np_rgb_img, bboxes)
            cv2.imwrite(cfg.TEST.DETECTED_IMAGE_PATH + file_name, image)

        with open(predict_result_path, "w") as f:
            for bbox in bboxes:
                coord = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_idx = int(bbox[5])
                class_name = CLASSES[class_idx]
                score = "%.4f" % score
                xmin, ymin, xmax, ymax = list(map(str, coord))
                bbox_mess = " ".join([class_name, score, xmin, ymin, xmax, ymax]) + "\n"
                f.write(bbox_mess)
                print("\t" + str(bbox_mess).strip())
