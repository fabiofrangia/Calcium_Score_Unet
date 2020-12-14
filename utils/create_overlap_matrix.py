import numpy as np 

class OverlapMatrix():
    def __init__(self, dim_x, dim_y, dim_z, patch_dim):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.patch_dim = patch_dim

    def create_matrix(self):

        matrix_3d = np.zeros((self.dim_x, self.dim_y, self.dim_z))
        for i in range(0, self.dim_x-self.patch_dim+1):
            for j in range(0,self.dim_y-self.patch_dim+1):
                for k in range(0,self.dim_z-self.patch_dim+1):
                    matrix_3d[i:i+self.patch_dim, j:j+self.patch_dim, k:k+self.patch_dim]+=1

        return matrix_3d



