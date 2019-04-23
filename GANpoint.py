import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

batch_size = 1
# noise truncation value
truncation_randomize = 0.5
truncation_walk = 0.05
#
scale_walk = 0.05

os.environ["TFHUB_CACHE_DIR"] = './models'
module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-512/1')

z = tf.Variable(np.zeros((batch_size, 128)).astype('f'))
y = tf.Variable(np.zeros((batch_size, 1000)).astype('f'))

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [1, 512, 512, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation_randomize))

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    run = True
    step = False
    behaviour = 'randomize'
    target = 'both'
    progress = tqdm()

    while True:

        key = cv2.waitKey(50)
        if key == 27:  # ESC: exit
            break
        elif key == 32:  # SPACE: toogle updating values or model, stop stepping
            run = not run
            step = False
        elif key == ord('s'):
            step = True
            run = True
        elif key == ord('r'):
            behaviour = 'randomize'
        elif key == ord('w'):
            behaviour = 'walk'
        elif key == ord('o'):
            behaviour = 'optimize'
        elif key == ord('y'):
            target = 'y'
        elif key == ord('z'):
            target = 'z'
        elif key == ord('b'):
            target = 'both'
        elif key == 43:  # +: reward
            print('Reward')
        elif key == 45:  #
            print('Punishment')

        if not run:
            continue

        if behaviour == 'randomize':
            if target in ['z', 'both']:
                z_np = np.random.normal(scale=truncation_randomize, size=(batch_size, 128)).astype('f')
            if target in ['y', 'both']:
                y_np = np.random.uniform(size=(batch_size, 1000)).astype('f')
                y_np = y_np / y_np.sum()  # sum of probabilities should be 1.0
        elif behaviour == 'walk':
            if target in ['z', 'both']:
                z_np += np.random.normal(scale=truncation_walk, size=(batch_size, 128)).astype('f')
            if target in ['y', 'both']:
                y_np += np.random.uniform(high=scale_walk, size=(batch_size, 1000)).astype('f')
                y_np[y_np < 0] = 0  # replace all negatives with zero
                y_np = y_np / y_np.sum()  # sum of probabilities should be 1.0
        elif behaviour == 'optimize':
            pass

        image = sess.run(samples, feed_dict={z: z_np, y: y_np})

        image = np.squeeze(image)
        image = (image + 1) / 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # red and blue have to be swaped

        cv2.imshow('Visions at GANpoint', image)
        progress.update()

        if step:
            run = False
            step = False

