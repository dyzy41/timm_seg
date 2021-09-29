import torch
from networks.get_model import get_net
from torchsummary import summary
import time

if __name__ == '__main__':
    net = get_net('SegNet').cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    t1 = time.time()
    y = net(x)
    print(y.shape)
    print(time.time()-t1)

