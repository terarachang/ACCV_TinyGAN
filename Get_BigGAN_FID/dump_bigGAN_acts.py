import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan
import functools
import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm
from PIL import Image
import os

INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
ACTIVATION_DIM = 2048

TRUNCATION = 0.5

BATCH_SIZE = 50
SAMPLE_NUM = 5000

assert SAMPLE_NUM % BATCH_SIZE == 0

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


def truncated_normal(size, threshold=2): # [-threshold, threshold]
    values = TRUNCATION * truncnorm.rvs(-threshold, threshold, size=size)
    return values


def save_images(images, c_id, offset):
    out_dir = 'tmp'
    # denorm from [-1, 1] to [0, 255]
    images = np.clip(((images + 1) / 2.0) * 256, 0, 255)
    images = np.uint8(images)

    for i, ndarr in enumerate(images):
        img = Image.fromarray(ndarr)
        img.save('{}/{}_{}.png'.format(out_dir, c_id, offset+i)) # starts from 0.png


def main():
    # build graph
    y_index = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    #y_index = tf.random_uniform([BATCH_SIZE], maxval=1000, dtype=tf.int32)
    z = tf.placeholder(tf.float32, shape=[None, 128])
    module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-128/1')

    y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label
    
    samples = module(dict(y=y, z=z, truncation=TRUNCATION))
    act = inception_activations(samples)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    
    noises = truncated_normal(SAMPLE_NUM*128).reshape(SAMPLE_NUM, 128)
    np.save('noises.npy', noises)
    #noises = np.load('noises.npy')
    
    out = np.zeros([SAMPLE_NUM, ACTIVATION_DIM])
    for c_id in tqdm(range(1000)):
        for i in tqdm(range(0, SAMPLE_NUM, BATCH_SIZE)):
            #samp = sess.run(samples, {y_index: [c_id]*BATCH_SIZE, z: noises[i: i+BATCH_SIZE]})
            #save_images(samp, c_id, i)
            out[i: i+BATCH_SIZE] = sess.run(act, {y_index: [c_id]*BATCH_SIZE, z: noises[i: i+BATCH_SIZE]})
        
        np.save('stat_big/act_{}.npy'.format(c_id), out)
    
    sess.close()
        
main()
