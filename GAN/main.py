from generativeAdversarialNetworks import generativeAdversarialNetworks
import matplotlib.pyplot as plt
import os
import torch
import argparse


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser(description="Generative adversarial networks")
    parser.add_argument('--batchSize', type=int, help='batch size', default=64)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=1)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'],help='name of the dataset- mnist or fashion-mnist')
    parser.add_argument('--input_ImageSize', type=int, help='input image size', default=28) #Both mnist and fashion mnist have size =28*28 pixels
    parser.add_argument('--cuda_usage', type=bool, default=False, help='Run the model on gpu or cpu. Assign it true for gpu and false for cpu')
    parser.add_argument('--output_directory', type=str, help='Directory name to save the generated images', default='output_images',)
    parser.add_argument('--graph_directory', type=str, help='Directory to save all graphs', default='graph_directory',)
    parser.add_argument('--model_dir', metavar='DIR', help='path to save trained model', default='saved_models/')
    parser.add_argument('--genLearningRate', type=float, help='Learning rate of generator model', default=0.0002)
    parser.add_argument('--disLearningRate', type=float, help='Learning rate of discriminator model', default=0.0002)
    parser.add_argument('--beta1_momentum', type=float, help='Momentum - beta1', default=0.5,)
    parser.add_argument('--beta2_momentum', type=float, help='Momentum - beta2', default=0.888)
    parser.add_argument('--load_model_parameters', type=bool, default=True, help='Load pre-trained weights and parameters of previos model and load this model from that checkpoint')

    return parser.parse_args()

def create_directory(x):
    if not os.path.exists(x):
        os.makedirs(x)

if __name__ == '__main__':
    args = parse_args()
    print("Print all arguments--------------------------------------------------------")
    print(args)    

    if args is None:
        exit()
    try:
        assert args.batchSize >= 1
    except:
        print('Batch size has to be a positive integer')
    try:
        assert args.epoch >= 1
    except:
        print('Total number of epochs has to be a positive integer')
    create_directory(args.output_directory)
    create_directory(args.graph_directory)
    create_directory(args.model_dir)
    torch.backends.cudnn.benchmark = True
    generativeAdversarialNetworks1 = generativeAdversarialNetworks(args)
    # Start training
    generativeAdversarialNetworks1.train()
    print("Finished training the model-------------------------------------------------------")

    generativeAdversarialNetworks1.checkResults(args.epoch)
    print("Finished testing the model and generated the images-----------------------------------")



