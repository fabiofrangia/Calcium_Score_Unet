import numpy as np
import torch 

def generate_patch(img):

    size = 64 # patch size
    stride = 25 # patch stride
    patches = img.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)[0]
    img = patches.contiguous().view(-1, 64, 64, 64)

    return img