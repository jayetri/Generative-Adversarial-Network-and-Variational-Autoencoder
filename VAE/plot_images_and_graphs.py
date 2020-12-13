import torch
import torch.nn as nn
import numpy as np
import imageio
from torchvision import datasets
import os
import gzip
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy.misc


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



def loss_VariationalAutoencoder_plot(overview_of_training, path='overview_of_training.png'):
    x = range(len(overview_of_training['VariationalAutoencoder_loss']))

    y1 = overview_of_training['VariationalAutoencoder_loss']
    y2 = overview_of_training[' Kullback-Leibler_loss']
    y3 = overview_of_training['Reconstruction_loss']


    plt.subplot(2, 2, 1)
    plt.plot(x, y1)
    plt.xlabel('Iteration')
    plt.ylabel('VariationalAutoencoder_loss')
    plt.title('VariationalAutoencoder_loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x, y2)
    plt.xlabel('Iteration')
    plt.ylabel(' Kullback-Leibler_loss')
    plt.title(' Kullback-Leibler_loss')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, y3)
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction_loss')
    plt.title('Reconstruction_loss')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, y1, label='VariationalAutoencoder_loss')
    plt.plot(x, y2, label=' Kullback-Leibler_loss')
    plt.plot(x, y3, label='Reconstruction_loss')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.legend(loc=2)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path + '/loss.png')

    plt.savefig(path)

    plt.close()






