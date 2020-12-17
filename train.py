import os
from utils.dataset import CT_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import generate_patch
from matplotlib import pyplot as plt
from model.unet_model import UNet
import tqdm
import torch.nn as nn
from torch import optim
from model.loss import DiceLoss
import torch

DIR_IMG = 'data/'
DIR_CHECKPOINT = 'checkpoints/'


def train_net(batch_size = 1, lr = 0.001, train_split = 0.5, epochs=20):

    """ Train function for UNET
    Args: 
        batch_size
        learning rate = lr
    After loading ct and associated mask, train the network with the UNET models.
    Train/Val split ratio is defined in the CT_Datased method

    Using Tensorboard it's also possbile to see a bung of stuff (TBD)
    $ tensorboard --logdir=runs
    """

    dataset = CT_Dataset(DIR_IMG)
    train, val = dataset.transform(train_split)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    net = UNet(1, 1, bilinear=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = DiceLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    epoch_loss = 0
    '''for ep in range(0, epochs):
        for i in train_loader:
            img = i['ct']['data'][0]
            mask = i['segm']['data'][0]
            print("Loaded")
            patch_img = generate_patch(img)
            patch_mask = generate_patch(mask)


            for j in range(0,(len(patch_img)-1)//4):
                mask_pred = net(patch_img.unsqueeze(1)[j*4:(j+1)*4].to(device))  

                loss = criterion(mask_pred.to(device), patch_mask[j*4:(j+1)*4].to(device).long())   
                print(loss.item())  
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if j%20==0:
                    plt.imshow(patch_img[j][:,:, 20],cmap='gray')
                    plt.title("img")
                    plt.show()
                    plt.imshow(mask_pred.cpu().detach().numpy()[0][0][:,:, 20],cmap='gray')
                    plt.title("Output")
                    plt.show()'''


if __name__ == '__main__':

    train_net()