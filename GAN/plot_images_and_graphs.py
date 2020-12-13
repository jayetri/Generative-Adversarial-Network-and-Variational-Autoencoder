import numpy as np
import imageio
from torchvision import datasets
import os
import gzip
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy.misc
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    if dataset == 'mnist':
        if not os.path.exists('saved_models' + '/' + 'mnist' ):
                os.makedirs('saved_models' + '/' + 'mnist')
        new_model_dir = os.path.join('saved_models', 'mnist')

        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        if not os.path.exists('saved_models' + '/' + 'fashion-mnist' ):
                os.makedirs('saved_models' + '/' + 'fashion-mnist')
        new_model_dir = os.path.join('saved_models', 'fashion-mnist')
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader, new_model_dir

def SaveAllImages(images, size, image_path):
    return func_to_save(images, size, image_path)

def func_to_save(images, size, path):
    #This is the step to merge_image
    height1, width1 = images.shape[1], images.shape[2]
    if images.shape[3]==1:
        img = np.zeros((height1 * size[0], width1 * size[1]))
        for index, image in enumerate(images):
            a1 = index % size[1]
            b1 = index // size[1]
            img[b1 * height1:b1 * height1 + height1, a1 * width1:a1 * width1 + width1] = image[:,:,0]
        merged_image = img

    elif (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((height1 * size[0], width1 * size[1], c))
        for index, image in enumerate(images):
            a1 = index % size[1]
            b1 = index // size[1]
            img[b1 * height1:b1 * height1 + height1, a1 * width1:a1 * width1 + width1, :] = image
        merged_image = img       
    else:
        raise ValueError('(images,size) does not have the right dimension')

    #finished_merging
    image = np.squeeze(merged_image)
    imageio.imwrite(path, image)
