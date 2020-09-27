import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-128/1')

# Sample random noise (z) and ImageNet label (y) inputs.
batch_size = 64
num_batches = int(1e6) // batch_size
out_dir = 'out_img'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

truncation = 0.5  # scalar truncation value in [0.0, 1.0]
z = truncation * tf.truncated_normal([batch_size, 128])  # noise sample
y_index = tf.random_uniform([batch_size], maxval=1000, dtype=tf.int32)
y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 128, 128, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation))

def save_images(images, b_id):
    # denorm from [-1, 1] to [0, 255]
    images = np.clip(((images + 1) / 2.0) * 256, 0, 255)
    images = np.uint8(images)
    
    for i, ndarr in enumerate(images):
        img = Image.fromarray(ndarr)
        img.save('{}/{}.png'.format(out_dir, b_id*batch_size+i)) # starts from 0.png


all_labels = np.zeros((num_batches*batch_size), dtype=int)
all_noises = np.zeros((num_batches*batch_size, 128))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    for b_id in tqdm(range(num_batches)):
        labels, noises, imgs = sess.run([y_index, z, samples]) # imgs: (bs, 128, 128, 3)
        all_labels[b_id*batch_size: (b_id+1)*batch_size] = labels
        all_noises[b_id*batch_size: (b_id+1)*batch_size] = noises
        save_images(imgs, b_id)
        
        if (b_id+1) % 100 == 0: # checkpoint
            np.save('all_labels.npy', all_labels)
            np.save('all_noises.npy', all_noises)
            print('Batch_{} is saved !'.format(b_id))
            
np.save('all_labels.npy', all_labels)