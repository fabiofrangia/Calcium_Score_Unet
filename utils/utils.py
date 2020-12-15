import numpy as np
import torch 

def generate_patch(img, size=64, stride=16):
    """ Method for the generation of patch
    Args:
        size: size of the pathc (cubic). Default is 64
        stride: (int). Default is 25
    """

    img[0][img[0]<0]=0
    shape_x = img[0].shape[0]
    shape_y = img[0].shape[1]
    shape_z = img[0].shape[2]
    img_patch = []
    for i in range(0, int(shape_x//stride)- int(size//stride)):
        for j in range(0, int(shape_y//stride)- int(size//stride)):
            for k in range(0, int(shape_z//stride)- int(size//stride)):
                patch_img = img[0][i*stride:i*stride+size, j*stride:j*stride+size, k*stride:k*stride+size]
                img_patch.append(patch_img.unsqueeze(0))

    img_patch = torch.cat(img_patch)
    return img_patch