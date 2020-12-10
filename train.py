import os
from utils.dataset import CT_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
DIR_IMG = 'data/'
DIR_CHECKPOINT = 'checkpoints/'

def train_net(batch_size = 1, lr = 0.001):

    dataset = CT_Dataset(DIR_IMG)
    train, val = dataset.transform(0.5)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')


if __name__ == '__main__':

    train_net()