##Cyclic-GAN
# X - paintings and Y - photographs
# Y' = F(x)
# X' = G(X)

batch_size = int(1)
import torch

# #Uncomment for using with Google Colab
# #Install the necessary dependencies
# # http://pytorch.org/
# from os.path import exists
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
# accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

# !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
# import torch
# !pip install Pillow==4.0.0
# !pip install PIL
# !pip install image


#Check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


# #Uncomment for using with Google Colab
# #Mount drive
# ##To load dataset from Drive
# from google.colab import drive
# drive.mount('/content/drive')
# #Check if drive is mounted
# # After executing the cell above, Drive
# # files will be present in "/content/drive/My Drive".
# !ls "/content/drive/My Drive"

print('#################################|Initializing|#################################')

# Create class for dataset
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform


class monet_dataset(Dataset):
    """Monet dataset."""

    def __init__(self, path_X, path_Y, transform=None):
        """
        Args:
            path_X (string): Path to the X images directory.
            path_Y (string): Path to the Y images directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Ximages_path = path_X
        self.Yimages_path = path_Y
        self.transform = transform
        self.list_X =  os.listdir(path_X)
        self.list_Y = os.listdir(path_Y)

    def __len__(self):
        return max(len(self.list_X),len(self.list_Y))

    def __getitem__(self, idx):
        img_name_X = os.path.join(self.Ximages_path,
                                self.list_X[idx%len(self.list_X)])
        imageY = io.imread(img_name_X)
        img_name_Y = os.path.join(self.Yimages_path,
                                self.list_Y[idx%len(self.list_Y)])
        imageX = io.imread(img_name_Y)

        if self.transform:
            imageX = self.transform(imageX)
            imageY = self.transform(imageY)

        return imageX,imageY


# create dataloader instances
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 #       transforms.RandomSizedCrop(224),
 #       transforms.RandomHorizontalFlip(),
 #       transforms.ToTensor(),
 #       transforms.Normalize(mean=[0.485, 0.456, 0.406],
 #                         std=[0.229, 0.224, 0.225]),
    ])

# Create datasets
trainset = monet_dataset('./../monet2photo/TrainA/trainA','./../monet2photo/TrainB/trainB', data_transform)
test_set = monet_dataset('./../monet2photo/TestA/testA','./../monet2photo/TestB/testB', data_transform)


# Functions to save and display image
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):

    img = img.cpu()
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imsave(a,img):

    img = img[0].cpu()
    npimg = img.detach().numpy()
    io.imsave('Images/' + str(a) + '.jpg', np.transpose(npimg, (1, 2, 0)))


#Discriminator model
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 4, stride= 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size = 14, stride= 1),  #Apply conv to get one dimensional output
        )

    def forward(self, x):

        x = self.block(x);
        return x


#Generator model
class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
        ## input > ENCODER > RESNET nine times > DECODER > output
        self.encode = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size = (7), stride= 1, padding = 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = (3), stride= 2, padding = 0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = (3), stride= 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )

        self.resnet = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3), stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = (3), stride= 1, padding=1),
            nn.BatchNorm2d(256),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride= 2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride= 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size = (7), stride= 1),
            nn.InstanceNorm2d(3),
            #nn.ReLU(),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encode(x);
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = F.relu(x + self.resnet(x));
        x = self.decode(x);

        return x


#Create an instances of the generator and discriminator and optimizers
import torch.optim as optim

F_gen = Generator()
F_dis = Discriminator()
G_gen = Generator()
G_dis = Discriminator()

criteriondis = nn.MSELoss()
criterion_feature = nn.L1Loss()
F_gen_optimizer = optim.Adam(F_gen.parameters(), lr=0.0002, betas = (0.5,0.999))
F_dis_optimizer = optim.Adam(F_dis.parameters(), lr = 0.0002, betas = (0.5,0.999))
G_gen_optimizer = optim.Adam(G_gen.parameters(), lr=0.0002, betas = (0.5,0.999))
G_dis_optimizer = optim.Adam(G_dis.parameters(), lr = 0.0002, betas = (0.5,0.999))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

F_gen.apply(weights_init)
G_gen.apply(weights_init)
F_dis.apply(weights_init)
F_dis.apply(weights_init)

G_gen = G_gen.cuda()
G_dis = G_dis.cuda()
F_gen = F_gen.cuda()
F_dis = F_dis.cuda()

try:
    #Load previously saved model
    checkpoint = torch.load('./parameters/model2.tar')
    F_gen.load_state_dict(checkpoint['F_gen_dict'])
    F_dis.load_state_dict(checkpoint['F_dis_dict'])
    G_gen.load_state_dict(checkpoint['G_gen_dict'])
    G_dis.load_state_dict(checkpoint['G_dis_dict'])
    F_gen_optimizer.load_state_dict(checkpoint['F_gen_optimizer_dict'])
    F_dis_optimizer.load_state_dict(checkpoint['F_dis_optimizer_dict'])
    G_gen_optimizer.load_state_dict(checkpoint['G_gen_optimizer_dict'])
    G_dis_optimizer.load_state_dict(checkpoint['G_dis_optimizer_dict'])
    print("|Loaded saved model|")
except:
    print("################||Training new model||###################")


G_gen = G_gen.cuda()
G_dis = G_dis.cuda()
F_gen = F_gen.cuda()
F_dis = F_dis.cuda()

F_gen.train()
F_dis.train()
G_gen.train()
G_dis.train()

real_images_Y = torch.randn(batch_size,3,256,256)
real_images_X = torch.randn(batch_size,3,256,256)

F_gen_loss_buffer = np.empty((1,1))
G_gen_loss_buffer = np.empty((1,1))
F_dis_loss_buffer = np.empty((1,1))
G_dis_loss_buffer = np.empty((1,1))
cyclic_loss_buffer = np.empty((1,1))
identity_loss_buffer = np.empty((1,1))

try:
    F_gen_loss_buffer = np.load('./parameters/F_gen_loss_buffer.npy')
    G_gen_loss_buffer = np.load('./parameters/G_gen_loss_buffer.npy')
    F_dis_loss_buffer = np.load('./parameters/F_dis_loss_buffer.npy')
    G_dis_loss_buffer = np.load('./parameters/G_dis_loss_buffer.npy')
    cyclic_loss_buffer = np.load('./parameters/cyclic_loss_buffer.npy')
    identity_loss_buffer = np.load('./parameters/identity_loss_buffer.npy')
except:
    print('could not load buffers..created new buffers')


print('################|Completed Initializations. Training started|################')


#Train the C-GAN
for epoch in range(100):

    F_running_dis_loss = 0
    F_running_loss_gen = 0
    G_running_dis_loss = 0
    G_running_loss_gen = 0
    running_cyclic_loss = 0
    running_identity_loss = 0

    for i in range(len(trainset)):

        # get the inputs
        real_images_X[0], real_images_Y[0] = trainset[i]
        real_images_X = real_images_X.cuda()
        real_images_Y = real_images_Y.cuda()
        ones = torch.ones((batch_size,1,1,1))
        zeros = torch.zeros((batch_size,1,1,1))
        ones = ones.cuda()
        zeros = zeros.cuda()

        #F
        # zero the parameter gradients
        F_gen_optimizer.zero_grad()
        F_dis_optimizer.zero_grad()
        G_gen_optimizer.zero_grad()
        G_dis_optimizer.zero_grad()

        #Minimize discriminator loss against ones
        F_dis_real_output = F_dis(real_images_Y)
        F_dis_real_loss = criteriondis(F_dis_real_output.float(), ones.float())/2

        #Minimize discriminator loss against zeros
        F_gen_output = F_gen(real_images_X)
        F_dis_fake_output = F_dis(F_gen_output)
        F_dis_fake_loss = criteriondis(F_dis_fake_output.float(), zeros.float())/2
        F_dis_tot_loss = F_dis_fake_loss + F_dis_real_loss;
        F_dis_tot_loss.backward()
        F_dis_optimizer.step()

        #Minimize generator loss
        F_gen_output = F_gen(real_images_X)
        F_dis_fake_output = F_dis(F_gen_output)
        F_gen_loss = criteriondis(F_dis_fake_output.float(), ones.float())
        #F_gen_loss.backward()
        #F_gen_optimizer.step()

        #G
        #Minimize discriminator loss against ones
        G_dis_real_output = G_dis(real_images_X)
        G_dis_real_loss = criteriondis(G_dis_real_output.float(), ones.float())/2

        #Minimize discriminator loss against zeros
        G_gen_output = G_gen(real_images_Y)
        G_dis_fake_output = G_dis(G_gen_output)
        G_dis_fake_loss = criteriondis(G_dis_fake_output.float(), zeros.float())/2
        G_dis_tot_loss = G_dis_fake_loss + G_dis_real_loss;
        G_dis_tot_loss.backward()
        G_dis_optimizer.step()

        #Minimize generator loss
        G_gen_output = G_gen(real_images_Y)
        G_dis_fake_output = G_dis(G_gen_output)
        G_gen_loss = criteriondis(G_dis_fake_output.float(), ones.float())
        #G_gen_loss.backward()
        #G_gen_optimizer.step()

        # zero the parameter gradients
        F_gen_optimizer.zero_grad()
        G_gen_optimizer.zero_grad()


        #Cycle consistency Loss
        # X > Y > X
        F_gen_output = F_gen(real_images_X);
        G_output = G_gen(F_gen_output);
        cyclic_loss_2 = criterion_feature(G_output, real_images_X);

        # Y > X > Y
        G_gen_output = G_gen(real_images_Y);
        F_output = F_gen(G_gen_output);
        cyclic_loss_1 = criterion_feature(F_output, real_images_Y)
        tot_cyclic_loss = 10*(cyclic_loss_1 + cyclic_loss_2);
        #tot_cyclic_loss.backward()
        F_identity_loss = criterion_feature(F_gen(real_images_Y), real_images_Y)*5
        G_identity_loss = criterion_feature(G_gen(real_images_X), real_images_X)*5
        tot_identity_loss = F_identity_loss + G_identity_loss
        F_gen_loss_tot = F_gen_loss + tot_cyclic_loss + F_identity_loss
        G_gen_loss_tot = G_gen_loss + tot_cyclic_loss + G_identity_loss
        F_gen_loss_tot.backward(retain_graph = True)
        G_gen_loss_tot.backward(retain_graph = True)
        F_gen_optimizer.step();
        G_gen_optimizer.step();


        # print statistics
        F_running_dis_loss += F_dis_tot_loss.item()
        F_running_loss_gen  += F_gen_loss_tot.item()
        G_running_dis_loss += G_dis_tot_loss.item()
        G_running_loss_gen  += G_gen_loss_tot.item()
        running_cyclic_loss += tot_cyclic_loss.item()
        running_identity_loss += tot_identity_loss.item()

        i += 1;

        if i % 500 == 499:    # print every 2000 mini-batches
            print('Epoch: %d | No of images: %5d | Cyclic loss: %.3f | Identity Loss: %.3f' %
                  (epoch + 1, i + 1,running_cyclic_loss / 500, running_identity_loss))
            print('F_dis_loss: %.3f | F_gen_loss: %.3f | G_dis_loss: %.3f | G_gen_loss: %.3f' %
                  (F_running_dis_loss / 500,  F_running_loss_gen / 500, G_running_dis_loss / 500, G_running_loss_gen / 500))

            imsave('temp_gen',F_gen(real_images_X))
            imsave('temp_ip',real_images_X)
            F_gen_loss_buffer = np.append(F_gen_loss_buffer, F_running_loss_gen)
            G_gen_loss_buffer = np.append(G_gen_loss_buffer, G_running_loss_gen)
            F_dis_loss_buffer = np.append(F_dis_loss_buffer, F_running_dis_loss)
            G_dis_loss_buffer = np.append(G_dis_loss_buffer, G_running_dis_loss)
            cyclic_loss_buffer = np.append(cyclic_loss_buffer, running_cyclic_loss)
            identity_loss_buffer = np.append(identity_loss_buffer, running_identity_loss)
            F_running_dis_loss = 0
            F_running_loss_gen  = 0
            G_running_dis_loss = 0
            G_running_loss_gen  = 0
            running_cyclic_loss = 0
            running_identity_loss = 0
            #save loss buffers
            np.save('./parameters/F_gen_loss_buffer',F_gen_loss_buffer)
            np.save('./parameters/G_gen_loss_buffer',G_gen_loss_buffer)
            np.save('./parameters/F_dis_loss_buffer',F_dis_loss_buffer)
            np.save('./parameters/G_dis_loss_buffer',G_dis_loss_buffer)
            np.save('./parameters/cyclic_loss_buffer',cyclic_loss_buffer)
            np.save('./parameters/identity_loss_buffer',identity_loss_buffer)

    #Save new parameters and output image after every epoch
    torch.save({
        'F_gen_dict': F_gen.state_dict(),
        'F_dis_dict': F_dis.state_dict(),
        'G_gen_dict': G_gen.state_dict(),
        'G_dis_dict': G_dis.state_dict(),
        'F_gen_optimizer_dict':F_gen_optimizer.state_dict(),
        'F_dis_optimizer_dict':F_dis_optimizer.state_dict(),
        'G_gen_optimizer_dict':G_gen_optimizer.state_dict(),
        'G_dis_optimizer_dict':G_dis_optimizer.state_dict(),
    }, './parameters/model2.tar')
    imsave(epoch,F_gen(real_images_X))
    imsave(str(epoch) + '_ip',real_images_X)
    print('#############################################################################################')

print('###########################################|Model Trained|###########################################')


# #Uncomment to check for memory in google colab
# # memory footprint support libraries/code
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# !pip install gputil
# !pip install psutil
# !pip install humanize
# import psutil
# import humanize
# import os
# import GPUtil as GPU
# GPUs = GPU.getGPUs()
# # XX: only one GPU on Colab and isn’t guaranteed
# gpu = GPUs[0]
# def printm():
#  process = psutil.Process(os.getpid())
#  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
#  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
# printm()
