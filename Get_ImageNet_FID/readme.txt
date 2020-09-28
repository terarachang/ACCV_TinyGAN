In order to make a fair comparison, we use BigGAN's preprocessed data to calculate the mean & covariance for FID score
1. download https://github.com/ajbrock/BigGAN-PyTorch
2. download ImageNet1000 training set & untar it. The default root is data/ImageNet
3. bash scripts/utils/prepare_data.sh
4. wget http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz
5. put the two my_*.py files under BigGAN-PyTorch/, and run them to dump the statistics of ImageNet
