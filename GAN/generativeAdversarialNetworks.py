import plot_images_and_graphs
import torch
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from plot_images_and_graphs import dataloader
import matplotlib.pyplot as plt
import os

class generator(nn.Module):
    
    def __init__(self, input_dim=100, output_dim=1, input_ImageSize=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_ImageSize = input_ImageSize


        self.layer0 = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4)),
            nn.BatchNorm1d(128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
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
        x = self.layer2(x)
        x = x.view(-1, 128, (self.input_ImageSize // 4), (self.input_ImageSize // 4))
        x = self.layer3(x)

        return x

class discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_ImageSize=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_ImageSize = input_ImageSize

        self.layer_d0 = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 4, 2, 1),
            nn.LeakyReLU(0.3)
        )
  
        self.layer_d1 = nn.Sequential(
            nn.Conv2d(32, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
        )
        self.layer_d2 = nn.Sequential(
            nn.Linear(128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3)
        )
        self.layer_d3 = nn.Sequential(
            nn.Linear(512, self.output_dim),
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
        x = self.layer_d0(input)
        x = self.layer_d1(x)
        x = x.view(-1, 128 * (self.input_ImageSize // 4) * (self.input_ImageSize // 4))
        x = self.layer_d2(x)
        x = self.layer_d3(x)

        return x

class generativeAdversarialNetworks(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batchSize = args.batchSize
        self.graph_directory = args.graph_directory
        self.model_dir = args.model_dir
        self.output_directory = args.output_directory
        self.dataset = args.dataset
        self.cuda_usage = args.cuda_usage
        self.input_ImageSize = args.input_ImageSize
        self.dimension_of_noise = 62
        self.load_model_parameters = args.load_model_parameters
        self.genLearningRate = args.genLearningRate
        self.disLearningRate = args.disLearningRate
        self.beta1_momentum= args.beta1_momentum
        self.beta2_momentum= args.beta2_momentum




        # load dataset
        self.data_loader, self.new_model_dir = dataloader(self.dataset, self.input_ImageSize, self.batchSize)
        data = self.data_loader.__iter__().__next__()[0]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        self.GeneratorModel, self.DiscriminatorModel = self.obtain_model(data) 
        self.GeneratorModel_optimizer, self.DiscriminatorModel_optimizer = self.obtain_optimizer()
        
         
        if self.cuda_usage:
            self.GeneratorModel.cuda()
            self.DiscriminatorModel.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()


        # generate random noise
        self.sample_noise_ = torch.rand((self.batchSize, self.dimension_of_noise))
        if self.cuda_usage:
            self.sample_noise_ = self.sample_noise_.cuda()

    #start
    def train(self):
        self.overview_of_training = {}
        self.overview_of_training['Discriminator_loss'] = []
        self.overview_of_training['Generator_loss'] = []
        self.overview_of_training['time_for_every_epoch'] = []
        self.overview_of_training['TimeTotalTaken'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batchSize, 1), torch.zeros(self.batchSize, 1)
        if self.cuda_usage:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.DiscriminatorModel.train()
        print('training start!!')
        start_of_training_time = time.time()
        for epoch in range(self.epoch):
            self.GeneratorModel.train()
            epoch_start_of_training_time = time.time()
            for iteration, (x_, _) in enumerate(self.data_loader):
                if iteration == self.data_loader.dataset.__len__() // self.batchSize:
                    break
                
                #generate random noise
                noise_ = torch.rand((self.batchSize, self.dimension_of_noise))
                if self.cuda_usage:
                    x_, noise_ = x_.cuda(), noise_.cuda()

               
                # update Discriminator network
                self.DiscriminatorModel_optimizer.zero_grad()

                Real_Discriminator = self.DiscriminatorModel(x_)
                Real_Discriminator_loss = self.BCE_loss(Real_Discriminator, self.y_real_)

                GeneratorModel_ = self.GeneratorModel(noise_)
                D_fake = self.DiscriminatorModel(GeneratorModel_)
 
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                Discriminator_loss = Real_Discriminator_loss + D_fake_loss
                self.overview_of_training['Discriminator_loss'].append(Discriminator_loss.item())

                Discriminator_loss.backward()
                self.DiscriminatorModel_optimizer.step()



                # update Generator network
                self.GeneratorModel_optimizer.zero_grad()

                GeneratorModel_ = self.GeneratorModel(noise_)
                D_fake = self.DiscriminatorModel(GeneratorModel_)
                Generator_loss = self.BCE_loss(D_fake, self.y_real_)
                self.overview_of_training['Generator_loss'].append(Generator_loss.item())

                Generator_loss.backward()
                self.GeneratorModel_optimizer.step()



                if ((iteration + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Discriminator_loss: %.8f, Generator_loss: %.8f" %
                          ((epoch + 1), (iteration + 1), self.data_loader.dataset.__len__() // self.batchSize, Discriminator_loss.item(), Generator_loss.item()))

            elapsed = time.time() - epoch_start_of_training_time
            self.overview_of_training['time_for_every_epoch'].append(elapsed)
            with torch.no_grad():
                self.checkResults((epoch+1))



        self.overview_of_training['TimeTotalTaken'].append(time.time() - start_of_training_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.overview_of_training['time_for_every_epoch']),
              self.epoch, self.overview_of_training['TimeTotalTaken'][0]))

        print("Finished training...............................all results will be stored")

        self.save_model(epoch, self.new_model_dir)
        self.check_and_create_path(self.graph_directory, self.dataset, str(self.batchSize))
        
        loss_path = os.path.join(self.graph_directory, self.dataset, str(self.batchSize))
        self.loss_plot(self.overview_of_training, loss_path, str(self.batchSize))



    def checkResults(self, epoch):
        self.GeneratorModel.eval()

        self.check_and_create_path(self.output_directory, self.dataset, str(self.batchSize))

        TotalNoOfSampless = self.sample_num
        ff= np.sqrt(TotalNoOfSampless)
        DimensionIImageFRame = int(np.floor(ff))

        
        #Generate fixed noise
        samples = self.GeneratorModel(self.sample_noise_)
        
        if self.cuda_usage:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        plot_images_and_graphs.SaveAllImages(samples[:DimensionIImageFRame * DimensionIImageFRame, :, :, :], [DimensionIImageFRame, DimensionIImageFRame],
                          self.output_directory + '/' + self.dataset + '/' + str(self.batchSize) + '/' + str(self.batchSize) + '_epoch%03d' % epoch + '.png')


    def save_model(self, epoch, new_model_dir):
        
        save_dir1 = os.path.join(new_model_dir, 'generator_model')
        save_dir2 = os.path.join(new_model_dir, 'discriminator_model')

        if not os.path.exists(save_dir1):
            os.mkdir(save_dir1)
        if not os.path.exists(save_dir2):
            os.mkdir(save_dir2)
        torch.save(
            self.GeneratorModel.state_dict(),
            os.path.join(save_dir1,'GeneratorModelenerator.pth'))
        torch.save(
            self.DiscriminatorModel.state_dict(),
            os.path.join(save_dir2,'discriminator.pth'))
        
        print('Saved generator model at epoch='+  str(epoch) + ' in '+ save_dir1)
        print('Saved discriminator model at epoch='+  str(epoch) + ' in '+ save_dir2)
        
    def obtain_optimizer(self):

        generator_optimizer = optim.Adam(self.GeneratorModel.parameters(), lr= self.genLearningRate, betas= (self.beta1_momentum, self.beta2_momentum))
        DiscriminatorModel_optimizer = optim.Adam(self.DiscriminatorModel.parameters(), lr= self.disLearningRate, betas= (self.beta1_momentum, self.beta2_momentum))
        return generator_optimizer, DiscriminatorModel_optimizer

    

    def obtain_model(self, data):
        GeneratorModel = generator(input_dim=self.dimension_of_noise, output_dim=data.shape[1], input_ImageSize=self.input_ImageSize)
        DiscriminatorModel = discriminator(input_dim=data.shape[1], output_dim=1, input_ImageSize=self.input_ImageSize)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)

        if self.load_model_parameters:
            if self.dataset== 'mnist':
                genisFile = os.path.isfile(os.path.join(dir_path, 'saved_models/mnist/generator_model/generator.pth'))  
                disisFile = os.path.isfile(os.path.join(dir_path, 'saved_models/mnist/discriminator_model/discriminator.pth')) 

            else:
                genisFile = os.path.isfile(os.path.join(dir_path, 'saved_models/fashion-mnist/generator_model/generator.pth'))  
                disisFile = os.path.isfile(os.path.join(dir_path, 'saved_model/fashion-mnist/discriminator_model/discriminator.pth')) 
            if genisFile ==True and disisFile == True:
                if self.dataset== 'mnist':
                    print('model exists ...needs to load') 
                    GeneratorModel.load_state_dict(torch.load(os.path.join(dir_path,'saved_models/mnist/generator_model/generator.pth')))
                    DiscriminatorModel.load_state_dict(torch.load(os.path.join(dir_path,'saved_models/mnist/discriminator_model/discriminator.pth')))
                    print("==> Loading Models...")

                else:
                    print('model exists ...needs to load') 
                    GeneratorModel.load_state_dict(torch.load(os.path.join(dir_path,'saved_models/fashion-mnist/generator_model/generator.pth')))
                    DiscriminatorModel.load_state_dict(torch.load(os.path.join(dir_path,'saved_models/fashion-mnist/discriminator_model/discriminator.pth')))
                    print("==> Loading Models...")
        else:
            print('model does not exist') 
            print("==> Creating Models...")
        if self.cuda_usage:
            GeneratorModel.cuda()
            DiscriminatorModel.cuda()           
        return GeneratorModel, DiscriminatorModel
    
    def check_and_create_path(self, A,B,C):
            if not os.path.exists(A + '/' + B + '/' + C):
                os.makedirs(A + '/' + B + '/' + C)



    def loss_plot(self, hist, path = 'History_for_training.png', batch = ''):
        x = range(len(hist['Discriminator_loss']))
        
        y1 = hist['Discriminator_loss']
        y2 = hist['Generator_loss']
        plt.subplot(2, 2, 1)
        plt.plot(x, y1)
        plt.xlabel('Iteration')
        plt.ylabel('Discriminator Loss')

        plt.title('Discriminator')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(x, y2)
        plt.xlabel('Iteration')
        plt.ylabel('Generator Loss')

        plt.title('Generator')
        plt.grid(True)

        plt.subplot(2, 2, 3)

        plt.plot(x, y1, label='Discriminator loss')
        plt.plot(x, y2, label='Generator loss')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.legend(loc=1)
        plt.grid(True)

        print(path)
        path = os.path.join(path + '/loss.png')
        print(path)

        plt.savefig(path)

        plt.close()
