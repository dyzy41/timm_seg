import torch
from networks.common_func.hrnet import *




if __name__ == '__main__':
    net = hrnet50()
    # nrte = meca(64, 3)

    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y)