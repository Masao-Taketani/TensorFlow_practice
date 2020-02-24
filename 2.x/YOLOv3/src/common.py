import tensorflow as tf


Class BatchNormalization(tf.keras.BatchNormalization):

    def call(self,
             training=False):
        if not training:
            training = tf.constant(False)
        # tf.logical_and(x, y, name=None)
        ## returns the truth value of x AND y element-wise
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer,
                  filters_shape,
                  down_sample=False,
                  activate=True,
                  bn=True):

    if downsample:
        """ZeroPadding2D:
        adds zero rows and columns to the top, bottom,
        left and right of the image tensor.
        Tuple of tuples: padding on each side.
        ((top_pad, bottom_pad), (left_pad, right_pad))
        """
        input_layer = tf.keras.layers.ZeroPadding2D(
            ((1, 0), (1, 0))
        )(input_layer)
        padding = "valid"
        strides = 2
    else:
        padding = "same"
        strides = 1

        conv = tf.keras.Conv2D(filters=filters_shape[-1],
                               kernel_size=filters_shape[0],
                               strides=strides,
                               padding=padding,
                               use_bias=not bn,
                               kernel_regularizer=tf.keras.regularizer.l2(0.0005),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(input_layer)

        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)

        return conv

def residual_block(input_layer,
                   input_channel,
                   filter_num1,
                   filter_num2):
    short_cut = input_layer
    # depth-wise conv
    conv = convolutional(input_layer,
                         filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv,
                         filters_shape=(3, 3, filter_num1, filter_num2))
    # skip connection(element-wise sum)
    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    """
    outputs double size of the original img
    and interpolates between values by nearseat
    neighbor
    """
    return tf.image.resize(input_layer,
                          (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
                          method="nearest")
