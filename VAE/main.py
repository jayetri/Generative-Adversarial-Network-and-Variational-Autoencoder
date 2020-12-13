from VariationalAutoencoder import VariationalAutoencoder
import os
import argparse


"""parsing and configuration"""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Variational Autoencoders")
    parser.add_argument('--batchSize', type=int, help='batch size', default=64)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=10)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'],help='name of the dataset- mnist or fashion-mnist')
    parser.add_argument('--input_ImageSize', type=int, help='input image size', default=28) #Both mnist and fashion mnist have size =28*28 pixels
    parser.add_argument('--cuda_usage', type=bool, default=False, help='Run the model on gpu or cpu. Assign it true for gpu and false for cpu')
    parser.add_argument('--output_directory', type=str, help='Directory name to save the generated images', default='output_images',)
    parser.add_argument('--graph_directory', type=str, help='Directory where all plots for losses are stored', default='results')
    parser.add_argument('--LearningRate', type=float, help='Learning rate of VAE model', default=0.0002)
    parser.add_argument('--beta1_momentum', type=float, help='Momentum - beta1', default=0.5,)
    parser.add_argument('--beta2_momentum', type=float, help='Momentum - beta2', default=0.888)

    return parser.parse_args()

def create_directory(x):
    if not os.path.exists(x):
        os.makedirs(x)


if __name__ == '__main__':
    arguments = parse_arguments()
    print("Print all arguments--------------------------------------------------------")
    print(arguments)    

    if arguments is None:
        exit()
    try:
        assert arguments.batchSize >= 1
    except:
        print('Batch size has to be a positive integer')
    try:
        assert arguments.epoch >= 1
    except:
        print('Total number of epochs has to be a positive integer')
    create_directory(arguments.output_directory)
    create_directory(arguments.graph_directory)

    VariationalAutoencoder1 = VariationalAutoencoder(arguments)
    
    # Start training
    VariationalAutoencoder1.train()
    print("Finished training the model-------------------------------------------------------")

    VariationalAutoencoder1.checkResults(arguments.epoch)
    print("Finished testing the model -----------------------------------")
