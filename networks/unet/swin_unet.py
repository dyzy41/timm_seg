import torch.nn as nn
from networks.common_func.swinT.swin_transformer import SwinTransformer
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class SwinUnet(nn.Module):
    def __init__(self, in_c, num_class=2, img_size=224):
        super(SwinUnet, self).__init__()
        self.img_size = img_size
        self.swin = SwinTransformer(img_size=self.img_size, in_chans=in_c)
        # self.pe0 = PatchExpand((56, 56), 96)
        # self.pe1 = PatchExpand((28, 28), 192)
        # self.pe2 = PatchExpand((14, 14), 384)
        # self.pe3 = PatchExpand((7, 7), 768)

    def forward(self, x):
        x_out, x_downsample, feature_pyramid = self.swin(x)
        return x_out, x_downsample, feature_pyramid
