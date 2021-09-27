import torch
from networks import *
from torchsummary import summary
import time

if __name__ == '__main__':
    net = U_Net(3, 6).cuda()
    summary(net, (3, 512, 512))
    x = torch.randn(2, 3, 512, 512).cuda()
    t1 = time.time()
    y = net(x)
    print(time.time()-t1)
