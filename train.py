import os
from utils.dataset import CT_Dataset

DIR_IMG = 'data/'
DIR_CHECKPOINT = 'checkpoints/'

def train_net():

    dataset = CT_Dataset(DIR_IMG)
    dataset = dataset.transform(1)
    print('Dataset size:', dataset, 'subjects')

if __name__ == '__main__':

    train_net()