import numpy as np
import torch

# 1.加载txt  -->  numpy  -->  tensor

def  loadtxt_np( txtfile):
    data = np.loadtxt(txtfile)

    # print(type(data))
    # print(data.shape)

    return data

def  loadtxt_torch( txtfile):
    data = np.loadtxt(txtfile)
    data_tensor = torch.tensor(data).float()  # torch.tensor(data)  # torch.float64,即double类型。
    # print(type(data))
    # print(data.shape)

    return data_tensor

if __name__ == '__main__':
    txtfile = './10000.txt'
    dsaafd = loadtxt_my(txtfile)

    print(type(dsaafd))
    print(dsaafd.shape)



