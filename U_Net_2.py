# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from osgeo import gdal
from torch.nn import ConvTranspose2d


class Ren(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # ConvTransposed2d  1
        self.CT1 = nn.Sequential(
            ConvTranspose2d(512, 256, kernel_size=2, stride=2, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        # ConvTransposed2d  2
        self.CT2 = nn.Sequential(
            ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        # ConvTransposed2d  3
        self.CT3 = nn.Sequential(
            ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.trans = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.end = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # print(x.shape)
        out_conv1 = self.conv1(x)
        # print(out_conv1.shape)
        out = self.pool(out_conv1)
        out_conv2 = self.conv2(out)
        out = self.pool(out_conv2)
        out_conv3 = self.conv3(out)
        # print(out_conv3.shape)
        out = self.pool(out_conv3)
        out = self.trans(out)
        out_CT1 = self.CT1(out)
        # print(out_CT1.shape)
        out = torch.cat([out_conv3, out_CT1], dim=1)
        out_conv4 = self.conv4(out)
        out_CT2 = self.CT2(out_conv4)
        out = torch.cat([out_conv2, out_CT2], dim=1)
        out_conv5 = self.conv5(out)
        out_CT3 = self.CT3(out_conv5)
        out = torch.cat([out_conv1, out_CT3], dim=1)
        out = self.end(out)
        return out


if __name__ == '__main__':
    Ren = Ren()

    rimage = gdal.Open("E:/test2/a/10_2.tif")
    r = rimage.ReadAsArray()
    r = r / 1.0
    R = torch.FloatTensor(r)
    print(R.shape)
    output = Ren(R)
