import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

batch_size = 1
# noise truncation value
truncation_randomize = 0.5
truncation_randomize_momentum = 0.01
truncation_walk = 0.05
# scaing of class vector
scale_randomize_momentum = 0.01
scale_walk = 0.05
reward_ratio = 0.5

os.environ["TFHUB_CACHE_DIR"] = './models'
module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-512/1')

y_tf = tf.Variable(np.zeros((batch_size, 1000)).astype('f'))
z_tf = tf.Variable(np.zeros((batch_size, 128)).astype('f'))

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [1, 512, 512, 3] and range [-1, 1].
samples = module(dict(y=y_tf, z=z_tf, truncation=truncation_randomize))

sess = tf.Session()
ganpoints = []
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    run = True
    step = False
    behaviour = 'randomize'
    target = 'both'

    while True:

        reward = None
        key = cv2.waitKeyEx(50)
        # if key > 0:
        #     print('Key pressed: {}'.format(key))

        if key == 27:  # ESC: exit
            break
        elif key == 32:  # SPACE: toogle updating values or model, stop stepping
            run = not run
            step = False
        elif key == ord('s'):  # s: one step after another
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

        elif key == 13:  # ENTER: set point
            ganpoints.append((y, z))
            print('Set point #{}'.format(len(ganpoints)))
        elif key == 8:  # BACK: return to point
            print('Load point #{}'.format(len(ganpoints)))
            y, z = ganpoints.pop()

        elif key == 82:  # UP: reward
            reward = '-'
        elif key == 84:  # DOWN: punishment
            reward = '+'

        if not run:  # for pausing
            continue

        if behaviour == 'randomize':
            if target in ['y', 'both']:
                y = np.random.uniform(size=(batch_size, 1000)).astype('f')
                y = y / y.sum()  # sum of probabilities should be 1.0
            if target in ['z', 'both']:
                z = np.random.normal(scale=truncation_randomize, size=(batch_size, 128)).astype('f')
        elif behaviour == 'walk':
            if target in ['y', 'both']:
                y += np.random.uniform(high=scale_walk, size=(batch_size, 1000)).astype('f')
                y[y < 0] = 0  # replace all negatives with zero
                y = y / y.sum()  # sum of probabilities should be 1.0
            if target in ['z', 'both']:
                z += np.random.normal(scale=truncation_walk, size=(batch_size, 128)).astype('f')
                z = np.clip(z, -truncation_randomize*2, truncation_randomize*2)  # confine at +- sigma 2
        elif behaviour == 'optimize':
            pass

        image = sess.run(samples, feed_dict={y_tf: y, z_tf: z})

        image = np.squeeze(image)
        image = (image + 1) / 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # red and blue have to be swaped

        cv2.imshow('Visions at GANpoint', image)

        if step:
            run = False
            step = False

