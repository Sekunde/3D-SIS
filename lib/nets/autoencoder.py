from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from scipy import misc
import os
import math
from PIL import Image
import numpy as np
from tqdm import tqdm

class ScanNet(Dataset):
    """ScanNet Image Dataset"""
    def __init__(self, folder='/mnt/local_datasets/ScanNet/frames_square'):
        scenes = [scene for scene in os.listdir(folder) if os.path.isdir(os.path.join(folder, scene))]
        self.images = []
        for scene in scenes:
            images = os.listdir(os.path.join(folder, scene, 'color'))
            for image in images:
                self.images.append(os.path.join(folder, scene, 'color', image))
        self.img_transform = transforms.Compose([transforms.ToTensor()])

    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = misc.imread(self.images[idx])
        img = self.resize_crop_image(img, [328, 256])
        img = self.img_transform(img)
        return img
            
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(nn.Conv2d(3,  32, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2, 0))

        self.conv2 = nn.Sequential(nn.Conv2d(32,  64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2, 0))

        self.conv3 = nn.Sequential(nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2, 0))

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())

        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64,  32, kernel_size=2, stride=2, padding=0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU())

        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32,  3, kernel_size=2, stride=2, padding=0),
                                     nn.Sigmoid())

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def decode(self, out):
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        return out

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 328 * 256 * 3),
                                 x.view(-1, 328 * 256 * 3), size_average=False)
    return BCE

def parse_args():
    parser = argparse.ArgumentParser(description='ScanNet AE')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()
    trainloader = DataLoader(ScanNet(), 
                             batch_size=config.batch_size,
                             shuffle=False, 
                             num_workers=1)

    model = Autoencoder().cuda()
    model.load_state_dict(torch.load('../DATA/ScanNet_Image/autoencoder{}.pth'.format(2)))
    model.train()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    for epoch in range(2, config.epochs):
        for batch_idx, data in enumerate(trainloader):
            data = Variable(data).cuda()
            recon_batch  = model(data)
            loss = loss_function(recon_batch, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader),
                    loss.data[0] / len(data)))
        
        model.eval()
        save_image(recon_batch.cpu().data, '../DATA/ScanNet_Image/image_{}.png'.format(epoch))
        torch.save(model.state_dict(), '../DATA/ScanNet_Image/autoencoder{}.pth'.format(epoch))
        model.train()
