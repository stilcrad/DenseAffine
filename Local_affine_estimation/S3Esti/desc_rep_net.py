import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class ResBlock(nn.Module):
    def __init__(self, in_channel_, out_channel_, kernel_size_,
                 stride_=1, padding_=0):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel_, out_channel_,
                              kernel_size=kernel_size_, padding=padding_)
        self.bn = nn.BatchNorm2d(out_channel_)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out += x

        return out


class DescRepNet(nn.Module):
    def __init__(self, desc_len=64):
        super(DescRepNet, self).__init__()
        self.desc_len = desc_len

        # 224
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.block1_desc = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(2)
        # 112
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.block2_desc = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(2)
        # 56
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.block3_desc = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # 56
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 112
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 224, regression of score
        self.block_score = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.maxpool3 = nn.MaxPool2d(2)
        # 224, regression of desc
        self.block4_desc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.desc_len, kernel_size=3, padding=1),
        )
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(self.maxpool1(x1))
        x3 = self.block3(self.maxpool2(x2))
        y = self.block4(x3) + self.block2_2(x2)
        y = self.block5(y) + self.block1_2(x1)
        score = self.block_score(y)

        desc = self.block4_desc(x3)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        desc = self.upsample(desc)

        return score, desc

# if __name__ == '__main__':
#     net = DescNet()
#     input = torch.rand(1, 3, 224, 224, dtype=torch.float)
#     output = net(input)
#     print('111', output.size())
