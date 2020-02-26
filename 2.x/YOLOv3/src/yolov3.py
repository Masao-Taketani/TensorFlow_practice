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
