import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


os.environ["TFHUB_CACHE_DIR"] = './models'
module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-512/1')

batch_size = 1
truncation = 0.5  # scalar truncation value in [0.0, 1.0]

z_np = truncation * np.random.normal(size=(batch_size, 128)).astype('f')
z = tf.Variable(z_np)

# y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
# y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

y_np = np.zeros((batch_size, 1000)).astype('f')
y_np[0, 0] = 1.0
y = tf.Variable(y_np)

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [1, 512, 512, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation))


sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    progress = tqdm()
    while True:
        z_np = truncation * np.random.normal(size=(batch_size, 128)).astype('f')

        image = sess.run(samples, feed_dict={z: z_np})

        image = np.squeeze(image)
        image = (image + 1) / 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # red and blue have to be swaped

        key = cv2.waitKey(1)
        if key == 27:  # ESC: exit
            break
        elif key == 43:  # +: reward
            print('Reward')
        elif key == 45:  #
            print('Punishment')
        elif key == 32:
            while cv2.waitKey(100) != 32:  # spacebar: pause on image
                pass

        cv2.imshow('Visions', image)
        progress.update()
