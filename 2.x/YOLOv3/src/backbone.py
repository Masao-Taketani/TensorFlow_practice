import tensorflow as tf
import common as cmn


fil_size = 3

def darknet53(input_data):
    # in case of input size: [256, 256, 3]
    x = cmn.convolutional(input_data, (fil_size, fil_size, 3, 32))
    # output shape: [256, 256, 32]
    x = cmn.convolutional(x, (fil_size, fil_size, 32, 64), downsample=True)
    # output shape: [128, 128, 64]

    for i in range(1):
        x = cmn.residual_block(x, 64, 32, 64)
    # output shape: [128, 128, 64]

    x = cmn.convolutional(x, (fil_size, fil_size, 64, 128), downsample=True)
    # output shape: [64, 64, 128]

    for i in range(2):
        x = cmn.residual_block(x, 128, 64, 128)
    # output shape: [64, 64, 128]

    x = cmn.convolutional(x, (fil_size, fil_size, 128, 256), downsample=True)
    # output shape: [32, 32, 256]

    for i in range(8):
        x = cmn.residual_block(x, 256, 128, 256)
    # output shape: [32, 32, 256]
    scale1 = x

    x = cmn.convolutional(x, (fil_size, fil_size, 256, 512), downsample=True)
    # output shape: [16, 16, 512]

    for i in range(8):
        x = cmn.residual_block(x, 512, 256, 512)
    # output shape: [16, 16, 512]

    scale2 = x
    x = cmn.convolutional(x, (fil_size, fil_size, 512, 1024), downsample=True)
    # output shape: [8, 8, 1024]

    for i in range(4):
        scale3 = cmn.residual_block(x, 1024, 512, 1024)
    # output shape: [8, 8, 1024]

    return scale1, scale2, scale3
