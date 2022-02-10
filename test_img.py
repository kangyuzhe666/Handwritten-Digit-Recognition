from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import inception_v3
from nets import nets_factory
from preprocessing import preprocessing_factory
# import cv2
import numpy as np
import os

test_image_size = 224
batch_size = 1

checkpoint_path = './resnet_v1_0210/model.ckpt-200000'

slim = tf.contrib.slim

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        network_fn = nets_factory.get_network_fn(
            "resnet_v1_50",
            num_classes=11,
            is_training=False)

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            "resnet_v1_50",
            is_training=False)



        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        L = []
        for root, dirs, files in os.walk('./minist_rgb/1'):
            for file in files:
                # if os.path.splitext(file)[1] == '.jpeg':
                L.append(os.path.join(file))

        print(L)
        print(len(L))
        test_ids = L
        print(test_ids)
        tot = len(L)
        results = list()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            time_start = time.time()
            for idx in range(0, tot, batch_size):
                images = list()
                idx_end = min(tot, idx + batch_size)
                for i in range(idx, idx_end):
                    image_id = test_ids[i]
                    test_path = os.path.join('./minist_rgb/1', image_id)
                    image = open(test_path, 'rb').read()
                    image = tf.image.decode_jpeg(image, channels=3)
                    processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                    processed_image = sess.run(processed_image)
                    images.append(processed_image)
                images = np.array(images)
                predictions = sess.run(logits, feed_dict={tensor_input: images})
                max_index = np.argmax(predictions)
                print(max_index)


                for i in range(idx, idx_end):
                    print('{} {}'.format(image_id, predictions[i - idx].tolist()))
            time_total = time.time() - time_start
            print('total time: {}, total images: {}, average time: {}'.format(
                time_total, len(test_ids), time_total / len(test_ids)))


if __name__ == '__main__':
    tf.app.run()