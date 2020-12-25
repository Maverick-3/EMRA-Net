import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse


def conv_3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv_1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CALayer(nn.Module):
    def __init__(self, channels, r=4):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels//r, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//r , channels, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Res2NetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, scales=4, groups=1, norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv_1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv_3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv_1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class DehazeNet(nn.Module):
    def __init__(self, opt):
        super(DehazeNet, self).__init__()
        nChannel = opt.nChannel
        nFeat = opt.nFeat
        self.opt = opt
        n_dense = 6
        self.res_blocks = n_dense
        
        # dwt2
        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.IDWT = DWTInverse(wave='haar').cuda()

        # F-1
        self.conv_0 = nn.Sequential(
            nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
            nn.BatchNorm2d(nFeat),
            nn.ReLU(inplace=True)
        )

        # res2net layers
        modules_1 = []
        modules_2 = []
        modules_3 = []
        for _ in range(n_dense):
            modules_1.append(Res2NetBottleneck(inplanes=64, planes=64//4))
            modules_2.append(Res2NetBottleneck(inplanes=128, planes=128//4))
            modules_3.append(Res2NetBottleneck(inplanes=256, planes=256//4))
        self.refine_1 = nn.Sequential(*modules_1)
        self.refine_2 = nn.Sequential(*modules_2)
        self.refine_3 = nn.Sequential(*modules_3)

        self.part1_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # DMT2
        self.conv_DWT2 = nn.Sequential(
            nn.Conv2d(in_channels=64*4, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.part2_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.part2_conv2 = nn.Conv2d(64, 12, kernel_size=3, padding=1)

        # DMT3 
        self.conv_DWT3 = nn.Sequential(
            nn.Conv2d(in_channels=128*4, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.part3_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.part3_conv2 = nn.Conv2d(32, 12, kernel_size=3, padding=1)

        # ca
        self.ca_1 = CALayer(64)
        self.ca_2 = CALayer(128)
        self.ca_3 = CALayer(256)
        self.ca_cat = CALayer(9)

        # conv_final
        self.conv_final = nn.Conv2d(9, 3, kernel_size=3, padding=1)


    def transformer(self, DWT_yl, DWT_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DWT_yh[0][:, :, i, :])
        list_tensor.append(DWT_yl)
        return torch.cat(list_tensor, 1)

    def Itransformer(self, out):
        yh = []
        C = out.shape[1] // 4
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

    def forward(self, x):
        # F_1
        F_1 = self.conv_0(x)
        shape_1 = F_1.data.size()[2:4]
        F_1_tmp = F_1

        part_1 = self.refine_1(F_1_tmp)
        part_1 = torch.add(part_1, F_1_tmp)
        part_1 = self.ca_1(part_1)
        part_1 = self.part1_conv(part_1)
        part_1 = torch.add(part_1, x)

        # F_2
        DWT2_yl, DWT2_yh = self.DWT(F_1)
        DWT2 = self.transformer(DWT2_yl, DWT2_yh)
        F_2 = self.conv_DWT2(DWT2)
        shape_2 = F_2.data.size()[2:4]
        F_2_tmp = F_2

        part_2 = self.refine_2(F_2_tmp)
        part_2 = torch.add(part_2, F_2_tmp)
        part_2 = self.ca_2(part_2)
        part_2 = self.part2_conv1(part_2)
        part_2 = self.part2_conv2(part_2)
        part_2 = self.Itransformer(part_2)
        part_2 = self.IDWT(part_2)
        part_2 = F.upsample_bilinear(part_2, size=shape_1)
        part_2 = torch.add(part_2, x)

        # F_3
        DWT3_yl, DWT3_yh = self.DWT(F_2)
        DWT3 = self.transformer(DWT3_yl, DWT3_yh)
        F_3 = self.conv_DWT3(DWT3)

        part_3 = self.refine_3(F_3)
        part_3 = torch.add(part_3, F_3)
        part_3 = self.ca_3(part_3)
        part_3 = self.part3_conv1(part_3)
        part_3 = self.Itransformer(part_3)
        part_3 = self.IDWT(part_3)
        part_3 = F.upsample_bilinear(part_3, size=shape_2)
        part_3 = self.part3_conv2(part_3)
        part_3 = self.Itransformer(part_3)
        part_3 = self.IDWT(part_3)
        part_3 = F.upsample_bilinear(part_3, size=shape_1)
        part_3 = torch.add(part_3, x)

        # fusion
        out = torch.cat((part_1, part_2, part_3), 1)
        out = self.ca_cat(out)
        out = self.conv_final(out)
        out = torch.add(x, out)
        return out
