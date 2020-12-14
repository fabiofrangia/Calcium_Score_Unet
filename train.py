import os
from utils.dataset import CT_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import generate_patch
from matplotlib import pyplot as plt
from model.unet_model import UNet

DIR_IMG = 'data/'
DIR_CHECKPOINT = 'checkpoints/'

def train_net(batch_size = 1, lr = 0.001):

    dataset = CT_Dataset(DIR_IMG)
    train, val = dataset.transform(0.5)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    net = UNet(1, 1, bilinear=False)

    for i in train_loader:
        img = i['ct']['data'][0]
        mask = i['segm']['data'][0]
        patch_img = generate_patch(img)
        patch_mask = generate_patch(mask)
        print(patch_img.shape)
        print(patch_mask.shape)
        plt.imshow(patch_img[5][:,:, 20],cmap='gray')
        plt.show()

        print(patch_img.unsqueeze(1).shape)

        mask_pred = net(patch_img.unsqueeze(1)[0:1])       


if __name__ == '__main__':

    train_net()