from torch.utils import data
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from glob import glob
import pickle
import torch
import os
import numpy as np
import random

class My_Dataset(data.Dataset):
    def __init__(self, image_dir, z_path, label_path, mode, transform):
        self.image_dir = image_dir
        self.z_path = z_path
        self.label_path = label_path
        self.mode = mode
        self.transform = transform
        self.preprocess()

    def preprocess(self, num_test=5000):
        """ Preprocess the label file. """
        self.dataset = []
#        selec_labels =  set(np.load('bad_ids_dw.npy'))  #set(np.load('hier_label.npy'))
        for img_dir, z_path, label_path in zip(self.image_dir.split('+'), self.z_path.split('+'), self.label_path.split('+')):
            filenames = os.listdir(img_dir)
            noises = np.load(z_path)
            labels = np.load(label_path)
            
            for i, fn in enumerate(filenames):
                idx = int(fn.split('.')[0])
                if labels[idx] > 397: continue # not animals
#                if labels[idx] not in selec_labels: continue
                self.dataset.append([os.path.join(img_dir, fn), noises[idx], labels[idx]])
        
        random.seed(999)
        random.shuffle(self.dataset)
        self.dataset = self.dataset[num_test:] if self.mode == 'train' else self.dataset[:num_test]
        self.num_images = len(self.dataset)
        print('# In {} set, # of data: {}'.format(self.mode, self.num_images))
        
        print('Finished preprocessing the dataset...')


    def __getitem__(self, index):
        filename, z, label = self.dataset[index]
        image = Image.open(filename)

        return self.transform(image), torch.FloatTensor(z), torch.tensor(label)
        
    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Real_Dataset(data.Dataset):
    def __init__(self, real_dir, transform):
        self.transform = transform
        self.real_dir = real_dir
        self.preprocess()

    def preprocess(self):
        """ Preprocess the label file. """
        dir_names = os.listdir(self.real_dir)

        with open('class2idx.pkl', 'rb') as f:
            class2idx = pickle.load(f)

#        selec_labels =  set(np.load('bad_ids_dw.npy'))  #set(np.load('hier_label.npy'))
        
        self.dataset = []
        for dir_name in dir_names:
            label = class2idx[dir_name]
            if label > 397: continue
#            if label not in selec_labels: continue
            filenames = glob(os.path.join(self.real_dir, dir_name, '*.JPEG'))
            for fn in filenames:
                self.dataset.append([fn, label])
        
        self.num_images = len(self.dataset)
        print('# In Real set, # of data: {}'.format(self.num_images))
        
        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and a dummy label."""
        filename, label = self.dataset[index]
        image = Image.open(filename)

        return self.transform(image), torch.tensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Interpolate_Dataset(data.Dataset):
    def __init__(self, image_dir, z_path, label_path, transform):
        self.image_dir = image_dir
        self.z_path = z_path
        self.label_path = label_path
        self.transform = transform
        self.preprocess()

    def preprocess(self, num_test=5000):
        """ Preprocess the label file. """
        scores = np.load('gan/models/intra_fid_scores.npy')
        cherry_set = set([i for i, s in enumerate(scores) if s < 70]) 
        self.dataset = []
        for img_dir, z_path, label_path in zip(self.image_dir.split('+'), self.z_path.split('+'), self.label_path.split('+')):
            filenames = os.listdir(img_dir)
            noises = np.load(z_path)
            labels = np.load(label_path)
            
            for i, fn in enumerate(filenames):
                idx = int(fn.split('.')[0])
                if labels[idx] not in cherry_set: continue
                
                self.dataset.append([fn, noises[idx], labels[idx]])
        
        random.seed(999)
        random.shuffle(self.dataset)
        self.dataset = self.dataset[:num_test]
        self.num_images = len(self.dataset)
        print('# of data: {}'.format(self.num_images))
        
        print('Finished preprocessing the dataset...')


    def __getitem__(self, index):
        """Interpolate between the same catelog"""
        filename, z, label = self.dataset[index]
        filename2, z2, label2 = self.dataset[index+1]
        image = Image.open(os.path.join(self.image_dir, filename))
        image2 = Image.open(os.path.join(self.image_dir, filename2))
        
        image = torch.stack((self.transform(image), self.transform(image2)), dim=0)
        z = np.vstack([z, z2])
        label = [label, label2]

        return image, torch.FloatTensor(z), torch.tensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, z_path, label_path, mode, image_size=128, batch_size=16, num_workers=4, isReal=False, isInter=False):
    """Build and return a data loader."""
    transform = T.Compose([
#            T.Resize(image_size),                         
            T.ToTensor(),                                 # [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])

    if isReal:
        dataset = Real_Dataset(image_dir, transform=transform)
    elif isInter:
        dataset = Interpolate_Dataset(image_dir, z_path, label_path, transform=transform)
    else:
        dataset = My_Dataset(image_dir, z_path, label_path, mode, transform=transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last = (mode=='train'),
                                  num_workers=num_workers,
                                  pin_memory = True
                                  )
    return data_loader
