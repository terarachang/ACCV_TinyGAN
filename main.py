import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
        os.makedirs(os.path.join(config.result_dir, 'interpolate'))

    if config.mode == 'train':
        train_loader = get_loader(config.image_dir, config.z_path, config.label_path, 
                            'train', config.image_size, config.batch_size, config.num_workers)
        test_loader = get_loader(config.image_dir, config.z_path, config.label_path, 
                            'test', config.image_size, config.batch_size, config.num_workers)
        real_loader = get_loader(config.real_dir, None, None, 
                            'train', config.image_size, config.batch_size, config.num_workers, isReal=True)
        
        solver = Solver(train_loader, test_loader, real_loader, config)
        solver.train()

    else:
        config.batch_size = 200
        solver = Solver(None, None, None, config)

        if config.mode == 'test_intra_quick':
            solver.test_intra_fid_quick()
        elif config.mode == 'test_intra_all':
            solver.test_intra_fid_all()
        elif config.mode == 'test_inter':
            solver.test_inter_fid()
        elif config.mode == 'test_bad':
            solver.test_bad()
        elif config.mode == 'test_is':
            solver.test_inception()

        else:
            train_loader = get_loader(config.image_dir, config.z_path, config.label_path, 
                                'train', config.image_size, config.batch_size, config.num_workers)
            test_loader = get_loader(config.image_dir, config.z_path, config.label_path, 
                                'test', config.image_size, config.batch_size, config.num_workers)
            solver = Solver(train_loader, test_loader, None, config)
            
            if config.mode == 'test_big':
                print('-------Inception-------')
                solver.test_inception_big()
                print('----------FID----------')
                solver.test_inter_fid_big()
                solver.test_intra_fid_big_all()
            else:
                solver.test()
                print('Finish Testing, start interpolate ...')
                solver.result_dir = os.path.join(config.result_dir, 'interpolate')
                solver.test_loader = get_loader(config.image_dir, config.z_path, config.label_path, 
                                    'test', config.image_size, 1, config.num_workers, isInter=True)
                solver.test_interpolate()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=128, help='dimension of noise')
    parser.add_argument('--c_dim', type=int, default=128, help='dimension of class embedding')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=5, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=5, help='number of strided conv layers in D')
    parser.add_argument('--lambda_gan', type=float, default=1e-2, help='weight for adversarial loss')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_epoch', type=int, default=30, help='number of total iterations for training D')
    parser.add_argument('--lr_decay_start', type=int, default=300000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    parser.add_argument('--use_vgg', type=str2bool, default=True)

    # Test configuration.
    parser.add_argument('--test_epoch', type=int, default=30, help='test model from this step')
    parser.add_argument('--use_numpy_fid', type=str2bool, default=True)

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test_intra_quick', 'test_intra_all', 'test_inter', 'test_is', 'test_big', 'test_bad', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--real_dir', type=str, default='/ssd/tera/train_128') # /media/tera/ILSVRC2012/train_128
    parser.add_argument('--image_dir', type=str, default='../ImageNet_1000/out_img+/ssd/tera/ImageNet_1000_2/out_img')
    parser.add_argument('--z_path', type=str, default='../ImageNet_1000/all_noises.npy+/ssd/tera/ImageNet_1000_2/all_noises.npy')
    parser.add_argument('--label_path', type=str, default='../ImageNet_1000/all_labels.npy+/ssd/tera/ImageNet_1000_2/all_labels.npy')
    parser.add_argument('--save_dir', type=str, default='gan')
    parser.add_argument('--real_fid_stat_dir', type=str, default='Get_ImageNet_FID')
    parser.add_argument('--real_incep_stat_dir', type=str, default='/ssd/tera/stat_real') # stat_real

    # Step size.
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=10)

    config = parser.parse_args()
    config.log_dir = os.path.join(config.save_dir, 'logs')
    config.model_save_dir = os.path.join(config.save_dir, 'models')
    config.sample_dir = os.path.join(config.save_dir, 'samples')
    config.result_dir = os.path.join(config.save_dir, 'results')
    print(config)
    main(config)
