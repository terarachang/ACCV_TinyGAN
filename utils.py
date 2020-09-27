import torch
from model import *
import os
from scipy.stats import truncnorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(config):
    if config.mode == 'train':
        G = Generator(config.image_size, config.g_conv_dim, config.z_dim, config.c_dim, config.g_repeat_num)
        D = Discriminator(config.image_size, config.d_conv_dim, config.d_repeat_num)

        g_optimizer = torch.optim.Adam(G.parameters(), config.g_lr, [config.beta1, config.beta2])
        d_optimizer = torch.optim.Adam(D.parameters(), config.d_lr, [config.beta1, config.beta2])
        print_network(G, 'G')
        print_network(D, 'D')
            
        G.to(device)
        D.to(device)

        if config.resume_epoch: # Start training from scratch or resume training.
            restore_model(config.resume_epoch, config.model_save_dir, G, D, g_optimizer, d_optimizer)

        return G, D, g_optimizer, d_optimizer

    else: # only the trained generator is needed when inference
        G = Generator(config.image_size, config.g_conv_dim, config.z_dim, config.c_dim, config.g_repeat_num)
        restore_model(config.test_epoch, config.model_save_dir, G, None, None, None)
        G.to(device)
        G.eval()
        return G, None, None, None


def save_model(ckpt_dir, epoch, G, D, g_optimizer, d_optimizer):
    """Save the trained models and optimizers."""
    model = {
        'G': G.state_dict(),
        'D': D.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict()
    }
    path = os.path.join(ckpt_dir, 'model_{}.tar'.format(epoch))
    torch.save(model, path)
    print('Saved model checkpoints into {} !'.format(path))


def restore_model(resume_epoch, ckpt_dir, G, D, g_optimizer, d_optimizer):
    """Restore the trained models and optimizers."""
    path = os.path.join(ckpt_dir, 'model_{}.tar'.format(resume_epoch))
    checkpoint = torch.load(path, map_location=device)
    G.load_state_dict(checkpoint['G'])
    
    if D:
        D.load_state_dict(checkpoint['D'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    
    print('The trained models from {} are loaded !'.format(path))


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def print_network(model, name, file=None):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {:.1f} M, ({})".format(num_params/10**6, num_params))


def build_tensorboard(log_dir):
    """Build a tensorboard logger."""
    from logger import Logger
    logger = Logger(log_dir)
    return logger


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def truncated_normal(size, threshold=2): # [-threshold, threshold]
    values = 0.5 * truncnorm.rvs(-threshold, threshold, size=size)
    return values
