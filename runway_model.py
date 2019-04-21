from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import numpy as np
import tensorflow as tf

import runway

slim = tf.contrib.slim

style_image_size = 256
image_size = 256
style_square_crop = False
content_square_crop = False

sess = tf.InteractiveSession()
style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])

if style_square_crop:
    style_img_preprocessed = image_utils.center_crop_resize_image(
        style_img_ph, style_image_size)
else:
    style_img_preprocessed = image_utils.resize_image(
        style_img_ph, style_image_size)

content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])

if content_square_crop:
    content_img_preprocessed = image_utils.center_crop_resize_image(
        content_img_ph, image_size)
else:
    content_img_preprocessed = image_utils.resize_image(
        content_img_ph, image_size)

stylized_images, _, _, bottleneck_feat = build_model.build_model(
    content_img_preprocessed,
    style_img_preprocessed,
    trainable=False,
    is_training=False,
    inception_end_point='Mixed_6e',
    style_prediction_bottleneck=100,
    adds_losses=False)


@runway.setup
def setup():
    init_fn = slim.assign_from_checkpoint_fn(
        './arbitrary_style_transfer/model.ckpt',
        slim.get_variables_to_restore()
    )
    sess.run([tf.local_variables_initializer()])
    init_fn(sess)
    return sess

stylize_inputs = {
    'content_image': runway.image,
    'style_image': runway.image,
    'interpolation_weight': runway.number(default=0.5, min=0, max=1, step=0.1)
}

@runway.command('stylize', inputs=stylize_inputs, outputs={'image': runway.image})
def stylize(sess, inputs):
    content_img_np = np.array(inputs['content_image'])
    style_image_np = np.array(inputs['style_image'])
    interpolation_weight = inputs['interpolation_weight']
    identity_params = sess.run(bottleneck_feat, feed_dict={style_img_ph: content_img_np})
    style_params = sess.run(bottleneck_feat, feed_dict={style_img_ph: style_image_np})
    stylized_image_res = sess.run(
        stylized_images,
        feed_dict={
            bottleneck_feat: identity_params * (1 - interpolation_weight) + style_params * interpolation_weight,
            content_img_ph: content_img_np
        })
    out = (stylized_image_res*255.0).astype(np.uint8)
    return out[0]


if __name__ == '__main__':
    runway.run(port=8540)
