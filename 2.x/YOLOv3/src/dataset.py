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

    def bbox_iou(self, boxes1, boxes2):
        """
        shape of boxes1(gt)                   : (1, 4)
        shape of boxes2(anchors_with_3_ratios): (3, 4)
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        """
        use of [..., ] :
        (e.g)>>> test = array([[5, 4, 6, 2],
                               [2, 4, 2, 6],
                               [9, 5, 3, 4]])
            >>> test[...,2]
            array([6, 2, 3])
            >>> test[...,3]
            array([2, 6, 4])
            >>> test[...,2] * test[...,3]
            array([12, 12, 12])

        boxes area for each scale (w * h)
        shape of boxes1_area: (1, 1)
        shape of boxes2_area: (3, 1)
        """
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        """
        the below code indicates (top_left_x, top_height_y, bottom_x_right, bottom_height_y)
        boxes1[..., :2] - boxes1[..., 2:] * 0.5 part:
            [[cx, cy] - [w_gt / 2, h_gt / 2]] = [[cx - w_gt / 2, cy - h_gt / 2]]
        boxes1[..., :2] + boxes1[..., 2:] * 0.5 part:
            [[cx, cy] + [w_gt / 2, h_gt / 2]] = [[cx + w_gt / 2, cy + h_gt / 2]]
        np.concatenate(...) part:
            [[cx - w_gt / 2, cy - h_gt / 2, cx + w_gt / 2, cy + h_gt / 2]]

        shape of boxes1: (1, 4)
        """
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        """
        boxes2[..., :2] - boxes2[..., 2:] * 0.5 part:
            [[cx, cy] - [w1 / 2, h1 / 2],         [[cx - w1 / 2, cy - h1 / 2],
             [cx, cy] - [w2 / 2, h2 / 2],     =    [cx - w2 / 2, cy - h2 / 2],
             [cx, cy] - [w3 / 2, h3 / 2]]          [cx - w3 / 2, cy - h3 / 2]]
        boxes2[..., :2] + boxes2[..., 2:] * 0.5 part:
            [[cx, cy] + [w1 / 2, h1 / 2],         [[cx + w1 / 2, cy + h1 / 2],
             [cx, cy] + [w2 / 2, h2 / 2],     =    [cx + w2 / 2, cy + h2 / 2],
             [cx, cy] + [w3 / 2, h3 / 2]]          [cx + w3 / 2, cy + h3 / 2]]
        np.concatenate(...) part:
             [[cx - w1 / 2, cy - h1 / 2, cx + w1 / 2, cy + h1 / 2],
              [cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2],
              [cx - w3 / 2, cy - h3 / 2, cx + w3 / 2, cy + h3 / 2]]

        shape of boxes2: (3, 4)
        """
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        """
        below code obtains the left up point and the right down point of
        the intersection.
        (e.g.)
        >>> arr1 = np.array([[5, 1, 7, 2]])
        >>> arr2 = np.array([[ 0,  1,  2,  3],
                             [ 4,  5,  6,  7],
                             [ 8,  9, 10, 11]])

        >>> np.maximum(test1[..., :2], test2[..., :2])
            np.array([[5, 1],
                      [5, 5],
                      [8, 9]])

        >>> np.minimum(test1[..., 2:], test2[..., 2:])
            np.array([[2, 2],
                      [6, 2],
                      [7, 2]])

        shape of left_up & right_down: (3, 2)
        """
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        """
        calcuate the intersection width and height to obtain the area
        (e.g.)
        >>> left_up
            np.array([[5, 1],
                      [5, 5],
                      [8, 9]])
        >>> right_down
            np.array([[2, 2],
                      [6, 2],
                      [7, 2]])

        >>> right_down - left_up
            np.array([[-3,  1],
                      [ 1, -3],
                      [-1, -7]])
        >>> np.maximum(right_down - left_up, 0.0)
            np.array([[0., 1.],
                      [1., 0.],
                      [0., 0.]])
        """
        inter_section = np.maximum(right_down - left_up, 0.0)
        # intersection width * intersection height
        # shape of inter_area: (3, 1)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        """
        shape of boxes1_area: (1, 1)
        shape of boxes2_area: (3, 1)
        shape of inter_area : (3, 1)

        Thus it broadcasts boxes1_area into (3, 1)
        (e.g.)
        >>> boxes1_area
            np.array([[5]])
        (after broadcasting)
            np.array([[5],
                      [5],
                      [5]])
        """
        union_area = boxes1_area + boxes2_area - inter_area
        # shape of the return value: (3, 1)  (iou value for each scale)
        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        # label = [(...), (...), (...)] for each scale
        label = [np.zeros((self.train_output_sizes[i],
                           self.train_output_sizes[i],
                           self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        """
        bbox_xywh:
               [np.array([[0, 0, 0, 0],         np.array[[0, 0, 0, 0],
                          [0, 0, 0, 0],      ‥           [0, 0, 0, 0],
                               ：                              ：
                          [0, 0, 0, 0]])                 [0, 0, 0, 0]]]
        """
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            """
            label file format used in this repo:
                img_path, xmin, ymin, xmax, ymax, id, xmin, ymin, ..., id
                Thus
                    bbox_coor: [xmin, ymin, xmax, ymax]
                    (img_path is already skipped before the process)
                    bbox_class_id: [id]
            """
            bbox_coor = bbox[:4]
            bbox_class_id = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_id] = 1.0
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
            # bbox_xywh: (cx, cy, w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]],
                                        axis=-1)
            """
            shape
                bbox_xywh[np.newaxis, :]    : [1, 4]
                self.strides[:, np.newaxis] : [3, 1]
                bbox_xywh_scaled            : [3, 4] (properly sacaled bbox for each scale)
            """
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                # np.floor(): round down to smaller int
                # ??? why + 0.5 ???
                """
                anchors_xywh:
                    np.array([[np.floor(xi)+0.5, np.floor(yi)+0.5, rx(i), ry(i)],
                              [np.floor(xi)+0.5, np.floor(yi)+0.5, rx(i+1), ry(i+1)],
                              [np.floor(xi)+0.5, np.floor(yi)+0.5, rx(i+2), ry(i+2)]])

                shape of self.anchors:
                    np.array([[[rx0, ry0],
                               [rx1, ry1],
                               [rx2, ry2]],

                              [[rx3, ry3],
                               [rx4, ry4],
                               [rx5, ry5]]

                              [[rx6, ry6],
                               [rx7, ry7],
                               [rx8, ry8]]])
                """
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                """
                bbox_xywh_scaled[i][np.newaxis, :]:
                    take bbox_xywh_scaled for one scale and makes the data into 2d

                bbox_iou(gt, anchors_with_3_ratios)
                shape of gt                   : (1, 4)
                shape of anchors_with_3_ratios: (3, 4)
                """
                ious_for_each_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                """
                !start tracing from here next time!
                """
                iou.append(ious_for_each_scale)
                iou_mask = ious_for_each_scale > 0.3
                # if any of the obtained aspect ratios ovalaps with the label
                # over 0.3
                if np.any(iou_mask):
                    x_id, y_id = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][y_id, x_id, iou_mask, :] = 0
                    label[i][y_id, x_id, iou_mask, 0:4] = bbox_xywh
                    label[i][y_id, x_id, iou_mask, 4:5] = 1.0
                    label[i][y_id, x_id, iou_mask, 5:] = smooth_onehot

                    bbox_id = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_id, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
