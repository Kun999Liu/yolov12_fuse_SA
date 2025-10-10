import torch
import torch.nn as nn


class NDSI_Layer(nn.Module):
    def __init__(self, dimension=1):
        super(NDSI_Layer, self).__init__()
        self.d = dimension
        # self.Channel_all = 3
        self.Channel_all = 6
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        ndsi = []
        # 提取红光、近红外、绿光、短波红外波段
        red = x[:, 0, :, :]
        green = x[:, 1, :, :]
        blue = x[:, 2, :, :]
        nir = x[:, 3, :, :]
        # 计算NDSI
        ndsi1 = (red - green) / (red + green + 1e-8)
        ndsi1 = ndsi1.unsqueeze(1)
        ndsi.append(ndsi1)
        ndsi2 = (red - blue) / (red + blue + 1e-8)
        ndsi2 = ndsi2.unsqueeze(1)
        ndsi.append(ndsi2)
        ndsi3 = (red - nir) / (red + nir + 1e-8)
        ndsi3 = ndsi3.unsqueeze(1)
        ndsi.append(ndsi3)
        ndsi4 = (green - blue) / (green + blue + 1e-8)
        ndsi4 = ndsi4.unsqueeze(1)
        ndsi.append(ndsi4)
        ndsi5 = (green - nir) / (green + nir + 1e-8)
        ndsi5 = ndsi5.unsqueeze(1)
        ndsi.append(ndsi5)
        ndsi6 = (blue - nir) / (blue + nir + 1e-8)
        ndsi6 = ndsi6.unsqueeze(1)
        ndsi.append(ndsi6)
        return torch.cat(ndsi, self.d)
