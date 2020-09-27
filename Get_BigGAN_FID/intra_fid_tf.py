'''
Modified from https://github.com/tsc2017/Frechet-Inception-Distance &
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
'''

import tensorflow as tf
import os
import functools
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan


INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
ACTIVATION_DIM = 2048


def inception_activations(images, height=INCEPTION_DEFAULT_IMAGE_SIZE, width=INCEPTION_DEFAULT_IMAGE_SIZE, num_splits = 1):
    images = tf.image.resize_bilinear(images, [height, width])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = functional_ops.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = INCEPTION_FINAL_POOL),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


if __name__ == '__main__':

    with tf.Session() as sess:
        # build graph
        real_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'real_activations')
        fake_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'fake_activations')
        fid = tfgan.eval.frechet_classifier_distance_from_activations(real_acts, fake_acts)
        
        fid_scores = np.zeros(1000)
        for i in tqdm(range(1000)):
            real_act = np.load('stat_big/act_{}.npy'.format(i))
            fake_act = np.load('../BigGAN-PyTorch/stat_real/act_{}.npy'.format(i))
    
            fid_score = sess.run(fid, {real_acts: real_act, fake_acts: fake_act})

            print('[{}] FID: {}'.format(i, fid_score))
            fid_scores[i] = fid_score
            
        np.save('fid_scores_real_big.npy', fid_scores)

