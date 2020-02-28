import numpy
import src.common as cmn
import src.utils as utils
import src.backbone as backbone
from src.config import cfg

import tensorflow as tf


NUM_CLASSES = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


def YOLOv3(input_layer):
    scale1, scale2, scale3 = backbone.darknet53(input_layer)

    conv = cmn.convolutional(scale3, (1, 1, 1024, 512))
    conv = cmn.convolutional(conv, (3, 3, 512, 1024))
    conv = cmn.convolutional(conv, (1, 1, 1024, 512))
    conv = cmn.convolutional(conv, (3, 3, 512, 1024))
    conv = cmn.convolutional(conv, (1, 1, 1024, 512))
    # For Large scale objs
    conv_lobj_branch = cmn.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = cmn.convolutional(conv_lobj_branch,
                                   # 3: # of scales used for the anchors
                                   # 5: (tx, ty, tw, th, objectness score)
                                   (1, 1, 1024, 3 * (NUM_CLASSES + 5)),
                                   activate=False,
                                   bn=False)

    conv = cmn.convolutional(conv, (1, 1, 512, 256))
    conv = cmn.upsample(conv)
    # depth-wise concat
    # [x, y, 256] + [x, y, 512]
    conv = tf.concat([conv, scale2], axis=-1)
    conv = cmn.convolutional(conv, (1, 1, 768, 256))
    conv = cmn.convolutional(conv, (3, 3, 256, 512))
    conv = cmn.convolutional(conv, (1, 1, 512, 256))
    conv = cmn.convolutional(conv, (3, 3, 256, 512))
    conv = cmn.convolutional(conv, (1, 1, 512, 256))
    # For Mid scale objs
    conv_mobj_branch = cmn.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = cmn.convolutional(conv_mobj_branch,
                                   (1, 1, 512, 3*(NUM_CLASSES + 5)),
                                   activate=False,
                                   bn=False)

    conv = cmn.convolutional(conv, (1, 1, 256, 128))
    conv = cmn.upsample(conv)
    # depth-wise concat
    # [x, y, 128] + [x, y, 256]
    conv = tf.concat([conv, scale1], axis=-1)
    conv = cmn.convolutional(conv, (1, 1, 384, 128))
    conv = cmn.convolutional(conv, (3, 3, 128, 256))
    conv = cmn.convolutional(conv, (1, 1, 256, 128))
    conv = cmn.convolutional(conv, (3, 3, 128, 256))
    conv = cmn.convolutional(conv, (1, 1, 256, 128))
    # For Small scale objs
    conv_sobj_branch = cmn.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = cmn.convolutional(conv_sobj_branch,
                                   (1, 1, 256, 3*(NUM_CLASSES + 5)),
                                   activate=False,
                                   bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i=0):
    # tf.shape return a tensor
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output,
                             (batch_size,
                             output_size,
                             output_size,
                             3,
                             5 + NUM_CLASSES))

    conv_raw_txty = conv_output[:,:,:,:,0:2]
    conv_raw_twth = conv_output[:,:,:,:,2:4]
    conv_raw_conf = conv_output[:,:,:,:,4:5]
    conv_raw_cls_prob = conv_output[:,:,:,:,5:]

    """
    For cy:
        inside of tf.range(...):
            from [0,1,2,..., len(output_size)-1]
            to [[0], [1], [2], ..., [len(output_size)-1]]

        inside of tf.tile(...):
            from [[0], [1], [2], ..., [len(output_size)-1]]
            to [[0,0,...,0], [1,1,...,1], ...,]

        output shape:
            [len(output_size), len(output_size)]
    """
    cy = tf.tile(tf.range(output_size,
                         dtype=tf.int32)[:, tf.newaxis],
                [1, output_size])

    """
    For cx:
        inside of tf.range(...):
            from [0,1,2,..., len(output_size)-1]
            to [[0,1,2,..., len(output_size)-1]]

        inside of tf.tile(...):
            from [[0,1,2,..., len(output_size)-1]]
            to [[0,1,2,..., len(output_size)-1], [0,1,2,..., len(output_size)-1], ...,]

        output shape:
            [len(output_size), len(output_size)]
    """
    cx = tf.tile(tf.range(output_size,
                         dtype=tf.int32)[tf.newaxis, :],
                [output_size, 1])
    # 1st channel: grid of cx
    # 2nd channel: grid of cy
    cxcy_grid = tf.concat([cx[:, :, tf.newaxis],
                         cy[:, :, tf.newaxis]],
                        axis=-1)
    cxcy_grid = tf.tile(cxcy_grid[tf.newaxis, :, :, tf.newaxis, :],
                      [batch_size, 1, 1, 3, 1])
    cxcy_grid = tf.cast(cxcy_grid, tf.float32)

    """
    For x and y:
        bx = σ(tx) + cx
        by = σ(ty) + cy
    For w and h:
        bw = pw * exp(tw)
        bh = ph * exp(th)
    """
    """
    shape of pred_xy:
        [batch_size,
         output_size,
         output_size,
         num_of_priors_per_scale,
         len([x,y])]
    """                    ]
    pred_xy = (tf.sigmoid(conv_raw_txty) + cxcy_grid) * STRIDES[i]
    """
    ANCHORS (3,3,2):
        np.array([[[p1_x, p1_y],
                   [p2_x, p2_y],
                   [p3_x, p3_y]],
                  [[p4_x, p4_y],
                   [p5_x, p5_y],
                   [p6_x, p6_y]],
                  [[p7_x, p7_y],
                   [p8_x, p8_y],
                   [p9_x, p9_y]]])
    """
    pred_wh = (ANCHORS[i] * STRIDES[i]) * tf.exp(conv_raw_twth)
    # 1st channel: prediction of x
    # 2nd channel: prediction of y
    # 3rd channel: prediction of w
    # 4th channel: prediction of h
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # num of depths: 1
    pred_conf = tf.sigmoid(conv_raw_conf)
    # num of depths: class size
    pred_prob = tf.sigmoid(conv_raw_cls_prob)
    """
    shape of return tensor:
        [batch_size,
         output_size,
         output_size,
         num_of_priors_per_scale,
         len([x,y,w,h])+1(=conf)+c+class_size]
    """
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

### need to recheck this func
def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)


def compute_loss(pred, conv, label, bboxes, i=0):

    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASSES))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    ### need to check pred and label
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]
    
