import torch.nn as nn
import MinkowskiEngine as Me


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, sparse=False):
    """3x3 convolution with padding"""
    if sparse:
        return Me.MinkowskiConvolution(in_planes, out_planes, kernel_size=3, stride=stride, 
                                       dilation=dilation, bias=False, dimension=3)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d if not sparse else Me.MinkowskiInstanceNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, sparse=sparse)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) if not sparse else Me.MinkowskiReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, sparse=sparse)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SparseBasicBlock3D(BasicBlock3D):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, 
                         stride=stride, downsample=downsample, groups=groups,
                         base_width=base_width, dilation=dilation,
                         norm_layer=norm_layer, sparse=True)
