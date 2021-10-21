import torch.nn as nn
from .swin_transformer import SwinTransformer

class SwinFeaturePyramid(nn.Module):
    def __init__(self, in_c):
        super(SwinFeaturePyramid).__init__()
        self.swin = SwinTransformer(img_size=512, patch_size=16, in_chans=in_c)

    def forward(self, x):


        return 0
