import os, sys
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

import scipy.ndimage as ndimage
from scipy import linalg
import torch


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # copied from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
    """ d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)) """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ' \
           'adding %s to diagonal of cov estimates') % eps
        print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        print('wat')
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real  

    tr_covmean = np.trace(covmean) 

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--stat_dir_path', type=str, default='big')
    args = parser.parse_args()

    labels = np.load('hier_label.npy')
    labels.sort()
    
    for c_id in tqdm(labels[:10]):
        fake_act = np.load('stat_small/fake_act_{}.npy'.format(c_id))
        mu, sigma = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
        del fake_act
        real_act = np.load('/ssd/tera/stat_real/act_{}.npy'.format(c_id))
        real_mu, real_sigma = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
    
        fid_score = numpy_calculate_frechet_distance(mu, sigma, real_mu, real_sigma)
        print('[{}] FID: {:.3f}'.format(c_id, fid_score))

if __name__ == '__main__':
    main()
