import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import time 
import os
import plot_images_and_graphs
import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable



class Encoder_VAE(nn.Module):
    
    def __init__(self, dataset = 'mnist', input_ImageSize = 28):
        super(Encoder_VAE, self).__init__()
        self.dimension_of_noise = 62
        self.dimension_of_output = 1
        self.input_ImageSize = input_ImageSize
        self.dimension_of_input = 1

        self.layer0 = nn.Sequential(
            nn.Conv2d(self.dimension_of_input, 32, 4, 2, 1),
            nn.LeakyReLU(0.3))
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3))
        self.layer3 = nn.Sequential(
            nn.Linear(512, 2 * self.dimension_of_noise),
        )
        # Initialization of weights
        for mod_wt in self.modules():
            if isinstance(mod_wt, nn.Linear):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()
            elif isinstance(mod_wt, nn.ConvTranspose2d):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()
            elif isinstance(mod_wt, nn.Conv2d):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = x.view(-1, 128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4))
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class Decoder_VAE(nn.Module):
    
    def __init__(self, dataset='mnist',input_ImageSize=28):
        super(Decoder_VAE, self).__init__()
        self.input_ImageSize = input_ImageSize
        self.dimension_of_output = 1
        self.dimension_of_input = 62

        self.layer0 = nn.Sequential(
            nn.Linear(self.dimension_of_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.layer1 = nn.Sequential(    
            nn.Linear(512, 128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4)),
            nn.BatchNorm1d(128 * (self.input_ImageSize// 4) * (self.input_ImageSize // 4)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, self.dimension_of_output, 4, 2, 1),
            nn.Sigmoid(),
        )
        # Initialization of weights
        for mod_wt in self.modules():
            if isinstance(mod_wt, nn.Linear):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()
            elif isinstance(mod_wt, nn.ConvTranspose2d):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()
            elif isinstance(mod_wt, nn.Conv2d):
                mod_wt.weight.data.normal_(0, 0.02)
                mod_wt.bias.data.zero_()

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = x.view(-1, 128, (self.input_ImageSize // 4), (self.input_ImageSize // 4))
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class calculate_latent_function(torch.nn.Module):
    def __init__(self, Encoder_VAE, Decoder_VAE):
        super(calculate_latent_function, self).__init__()
        self.dimension_of_noise = 64
        self.Encoder_VAE = Encoder_VAE
        self.Decoder_VAE = Decoder_VAE

    def _Sampling_of_mean_and_sigma(self, encoding_hidden):
        #It returns latent normal sample ~ N(mu, sigma^2)

        mu = encoding_hidden[:, :62]
        logarithm_of_sigma = encoding_hidden[:, 62:]
        sigma = torch.exp(logarithm_of_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).type(torch.FloatTensor)
        self.noise_mean = mu
        self.noise_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        encoding_hidden = self.Encoder_VAE(state)
        z = self._Sampling_of_mean_and_sigma(encoding_hidden)
        return self.Decoder_VAE(z)

def latent_loss(noise_mean, noise_stddev):
    mean_sq = noise_mean * noise_mean
    stddev_sq = noise_stddev * noise_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class VariationalAutoencoder(object):
    def __init__(self, arguments):
        # parameters
        self.epoch = arguments.epoch
        self.number_of_samples = 64
        self.batchSize = arguments.batchSize
        self.output_directory = arguments.output_directory
        self.graph_directory = arguments.graph_directory
        self.dataset = arguments.dataset
        self.cuda_usage = arguments.cuda_usage
        self.LearningRate = arguments.LearningRate
        self.beta1_momentum = arguments.beta1_momentum
        self.beta2_momentum = arguments.beta2_momentum

        # networks init
        self.En = Encoder_VAE(self.dataset)
        self.De = Decoder_VAE(self.dataset)
        self.VariationalAutoencoder = calculate_latent_function(self.En, self.De)

        #Initializing the optimizer
        self.VariationalAutoencoder_optimizer = self.obtain_optimizer()


        if self.cuda_usage == False:
            self.BCE_loss = nn.BCELoss()

        else:
            self.VariationalAutoencoder.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            
        self.dimension_of_noise = 62

        #creating gaussian noise with mean=0 and variance = 1
        self.noise_normalized = np.random.normal(0, 1, (self.batchSize, self.dimension_of_noise)).astype(np.float32)

        # fixed noise
        if self.cuda_usage == False:
            self.sample_noise_ = Variable(torch.from_numpy(self.noise_normalized).type(torch.FloatTensor), volatile=True)
        else:
            self.sample_noise_ = Variable(torch.from_numpy(self.noise_normalized).type(torch.FloatTensor).cuda(), volatile=True)

            

        # load dataset
        
        if self.dataset == 'fashion-mnist':
            self.data_loader = DataLoader(
                datasets.FashionMNIST('data_stored/fashion-mnist', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=self.batchSize, shuffle=True)

        elif self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('data_stored/mnist', train=True, download=True,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                                           batch_size=self.batchSize, shuffle=True)

        

    def train(self):
        self.overview_of_training = {}
        self.overview_of_training['VariationalAutoencoder_loss'] = []
        self.overview_of_training[' Kullback-Leibler_loss'] = []
        self.overview_of_training['Reconstruction_loss'] = []
        self.overview_of_training['per_epoch_time'] = []
        self.overview_of_training['timeTotallyTaken'] = []

        self.VariationalAutoencoder.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.En.train()
            epoch_start_time = time.time()
            for iter, (inp_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batchSize:
                    break

                noise_ = torch.from_numpy(self.noise_normalized).type(torch.FloatTensor)

                if self.cuda_usage== False:
                    inp_, noise_ = Variable(inp_), Variable(noise_)

                else:
                    inp_, noise_ = Variable(inp_.cuda()), Variable(noise_.cuda())

                # update VariationalAutoencoder network

                a1 = self.VariationalAutoencoder(inp_)
                self.VariationalAutoencoder_optimizer.zero_grad()
                Kullback_Leibler_loss = latent_loss(self.VariationalAutoencoder.noise_mean, self.VariationalAutoencoder.noise_sigma)
                Reconstruction_loss = self.BCE_loss(a1, inp_)
                VariationalAutoencoder_loss = Reconstruction_loss +  Kullback_Leibler_loss
                
                self.overview_of_training['VariationalAutoencoder_loss'].append(VariationalAutoencoder_loss.item())
                self.overview_of_training[' Kullback-Leibler_loss'].append( Kullback_Leibler_loss.item())
                self.overview_of_training['Reconstruction_loss'].append(Reconstruction_loss.item())
                
                VariationalAutoencoder_loss.backward()
                self.VariationalAutoencoder_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f"%
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batchSize,
                           time.time() - start_time,VariationalAutoencoder_loss.item()))


                if np.mod((iter + 1), 300) == 0:
                    samples = self.De(self.sample_noise_)
                    if self.cuda_usage == False:
                        samples = samples.data.numpy().transpose(0, 2, 3, 1)
                    else:
                        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
                        
                    calculate_total_number_of_samples = min(self.number_of_samples, self.batchSize)
                    height_Manifold = int(np.floor(np.sqrt(calculate_total_number_of_samples)))
                    width_Manifold = int(np.floor(np.sqrt(calculate_total_number_of_samples)))

                    plot_images_and_graphs.SaveAllImages(samples[:height_Manifold * width_Manifold, :, :, :], [height_Manifold, width_Manifold],
                                self.create_directory(self.graph_directory + '/' + self.model_dir) + '/' + '_train_{:02d}_{:04d}.png'.format(epoch, (iter + 1)))

            self.overview_of_training['per_epoch_time'].append(time.time() - epoch_start_time)
            self.checkResults((epoch+1))

        self.overview_of_training['timeTotallyTaken'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.overview_of_training['per_epoch_time']),
              self.epoch, self.overview_of_training['timeTotallyTaken'][0]))
        print("Training finish!... save training results")

        self.save_trained_model()
        plot_images_and_graphs.loss_VariationalAutoencoder_plot(self.overview_of_training, os.path.join(self.output_directory, self.model_dir))

    def checkResults(self, epoch):
        self.De.eval()

        calculate_total_number_of_samples = min(self.number_of_samples, self.batchSize)
        c= np.sqrt(calculate_total_number_of_samples)
        image_frame_dim = int(np.floor(c))

        
        """ random noise """
        if self.cuda_usage:
            sample_noise_ = Variable(torch.from_numpy(self.noise_normalized).type(torch.FloatTensor).cuda(), volatile=True)
        else:
            sample_noise_ = Variable(torch.from_numpy(self.noise_normalized).type(torch.FloatTensor), volatile=True)

            samples = self.De(sample_noise_)

        if self.cuda_usage == False:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)


        plot_images_and_graphs.SaveAllImages(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.create_directory(self.graph_directory + '/' + self.model_dir) + '/' +
                            '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset,self.batchSize, self.dimension_of_noise)

    def save_trained_model(self):
        output_directory = os.path.join(self.output_directory, self.model_dir)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        torch.save(self.VariationalAutoencoder.state_dict(), os.path.join(output_directory + '_VariationalAutoencoder.pth'))

    def obtain_optimizer(self):    
        VAE_optimizer = optim.Adam(self.VariationalAutoencoder.parameters(),lr= self.LearningRate, betas=(self.beta1_momentum, self.beta2_momentum))
        return VAE_optimizer
    

    def create_directory(self, x):
        if not os.path.exists(x):
            os.makedirs(x)
        return x