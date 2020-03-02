import os
import cv2
import random
import numpy as np
import tensorflow as tf
import src.utils as utils
from src.config import cfg


class Dataset(object):

    def __init__(self, dataset_type):
        self.anno_path = cfg.TRAIN.ANNO_PATH if dataset_type == "train" \
                         else cfg.TEST.ANNO_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == "train" \
                           else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == "train" \
                          else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == "train" \
                        else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        with open(self.anno_path, "r") as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt \
                          if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    # if an object is requested to work as an iterator,
    # it calls __iter__() method
    def __iter__(self):
        return self

    def __next__(self):

        with tf.device("/cpu:0"):
            # ranodom.choice: pick an element randomly
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size,
                                    self.train_input_size,
                                    self.train_input_size,
                                    3),
                                    dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size,
                                          self.train_output_sizes[0],
                                          self.train_output_sizes[0],
                                          self.anchor_per_scale,
                                          5 + self.num_classes),
                                          dtype=np.float32)

            batch_label_mbbox = np.zeros((self.batch_size,
                                          self.train_output_sizes[1],
                                          self.train_output_sizes[1],
                                          self.anchor_per_scale,
                                          5 + self.num_classes),
                                          dtype=np.float32)

            batch_label_lbbox = np.zeros((self.batch_size,
                                          self.train_output_sizes[2],
                                          self.train_output_sizes[2],
                                          self.anchor_per_scale,
                                          5 + self.num_classes),
                                          dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size,
                                      self.max_bbox_per_scale,
                                      4),
                                      dtype=np.float32)

            batch_mbboxes = np.zeros((self.batch_size,
                                      self.max_bbox_per_scale,
                                      4),
                                      dtype=np.float32)

            batch_lbboxes = np.zeros((self.batch_size,
                                      self.max_bbox_per_scale,
                                      4),
                                      dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    idx = self.batch_count * self.batch_size + num
                    if idx >= self.num_samples:
                        idx -= self.num_samples
                    annotation = self.annotations[idx]
                    img, bboxes = self.parse_annotation(annotation)
                    label_bboxes, pred_bboxes = self.preprocess_true_boxes(bboxes)


    def parse_annotation(self, annotation):

        line = annotation.split()
        img_path = line[0]
        if not os.path.exists(img_path):
            raise KeyError("The image file %s does not exist!".format(img_path))
        img = cv2.imread(img_path)
        """
        usage: map(func or lambda_exp, sequence_obj)
        return: map obj
        (e.g.)
        original_list = list(range(10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mapped_list = map(lambda x: x**2, original_list)
        print(list(mapped_list)) # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        """
        bboxes = np.array([list(map(int, box.split(","))) for box in line[1:]])

        if self.data_aug:
            img, bboxes = self.random_horizontal_flip(np.copy(img),
                                                      np.copy(bboxes))
            img, bboxes = self.random_crop(np.copy(img),
                                           np.copy(bboxes))
            img, bboxes = self.random_translate(np.copy(img),
                                                np.copy(bboxes))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, bboxes = utils.image_preprocess(np.copy(img),
                                             [self.train_input_size,
                                             self.train_input_size],
                                             np.copy(bboxes))
        return img, bboxes

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i],
                           self.train_output_sizes[i],
                           self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # np.full: initialize ndarray with a specified number
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            delta = 0.01
            """
            if num_classes = 10 and onehot = [0.0, 1.0, ..., 0.0]:
                uniform_dist = [0.1, 0.1, ..., 0.1]
                smooth_onehot = [0.0, 0.99, ..., 0.0] + [0.001, 0.001, ..., 0.001]
                              = [0.001, 0.991, ..., 0.001]
            """
            smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution
            
