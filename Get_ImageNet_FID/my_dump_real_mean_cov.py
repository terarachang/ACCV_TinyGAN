''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan
import functools
import numpy as np
from tqdm import tqdm


def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=1,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser


def denorm(x):
  """Convert the range from [-1, 1] to [0, 1]."""
  out = (x + 1) / 2
  return out.clamp_(0, 1)



def run(config):
    # Get loader
    config['drop_last'] = False
    loaders = utils.get_data_loaders(**config)
    
    # build graph
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

    images_holder = tf.placeholder(tf.float32, [None, 128, 128, 3])
    activations = inception_activations(images_holder)
    real_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'real_activations')
    fake_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'fake_activations')
    fid = tfgan.eval.frechet_classifier_distance_from_activations(real_acts, fake_acts)
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    
    device = 'cuda'
    print('# data:', len(loaders[0].dataset))
    pool = np.zeros((len(loaders[0].dataset), 2048))
    
    offs = 0
    for i, (x, y) in enumerate(tqdm(loaders[0])):
        x = np.transpose(x.numpy(), (0, 2, 3, 1)) # [NCHW] -> [NHWC]
        pool[offs: offs+len(y)] = sess.run(activations, {images_holder: x}) 
        offs += len(y)
        
    # for FID
    mean, cov = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    np.savez('stat_real.npz', mean=mean, cov=cov)
    

def main():
    # parse command line    
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':    
    main()
