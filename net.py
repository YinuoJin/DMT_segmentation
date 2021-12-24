import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    """
    - : Conv
    \ : Maxpool
    / : Upsample (ConvTransposed)
    ->: Concatenate
    """

    def __init__(self, c_in=1, c_out=1, c_1=64, p=0.2):
        super(Unet, self).__init__()
        self.dropout_net = nn.Dropout2d(p=p)
        self.contract_net = Contraction(c_in, c_1, self.dropout_net)
        self.double_conv_net = DoubleConv2d(c_1 * 8, c_1 * 16, self.dropout_net)

        self.expand_net = Expansion(c_1 * 16, self.dropout_net)
        self.out_net = nn.Sequential(
            nn.Conv2d(c_1, c_out, kernel_size=1),
            self.activate_net(c_out)
        )

    def forward(self, x):
        x, conv_output = self.contract_net(x)
        x = self.double_conv_net(x)
        x = self.expand_net(x, conv_output)
        x = self.out_net(x)

        return x

    def activate_net(self, c_out):
        if c_out == 1:
            return nn.Sigmoid()
        elif c_out == 2:
            norm_vec = NormVectorActivation()
            return norm_vec
        else:
            return nn.Softmax(dim=1)



class ResUnet(Unet):
    """
    Modified Unet architecture with Residual block in each DoubleConv
    """

    def __init__(self, c_in=1, c_out=1, c_1=64, p=0.2):
        super(ResUnet, self).__init__(c_in, c_out, c_1, p)
        self.contract_net = ResContraction(c_in, c_1, self.dropout_net)
        self.double_conv_net = ResDoubleConv2d(c_1 * 8, c_1 * 16, self.dropout_net)
        self.expand_net = ResExpansion(c_1 * 16, self.dropout_net)


class Contraction(nn.Module):
    """
    Contracting path of Unet (Encoder)
    """

    def __init__(self, c_in, c_1, dropout_net):
        super(Contraction, self).__init__()
        self.dconv_net1 = DoubleConv2d(c_in, c_1, dropout_net)
        self.dconv_net2 = DoubleConv2d(c_1, c_1 * 2, dropout_net)
        self.dconv_net3 = DoubleConv2d(c_1 * 2, c_1 * 4, dropout_net)
        self.dconv_net4 = DoubleConv2d(c_1 * 4, c_1 * 8, dropout_net)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.dconv_net1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_net2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_net3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_net4(x)
        x = self.maxpool(conv4)

        return x, (conv1, conv2, conv3, conv4)


class ResContraction(Contraction):
    """
    Contracting path of ResUnet (Encoder)
    """

    def __init__(self, c_in, c_1, dropout_net):
        super(ResContraction, self).__init__(c_in, c_1, dropout_net)
        self.dconv_net1 = ResDoubleConv2d(c_in, c_1, dropout_net)
        self.dconv_net2 = ResDoubleConv2d(c_1, c_1 * 2, dropout_net)
        self.dconv_net3 = ResDoubleConv2d(c_1 * 2, c_1 * 4, dropout_net)
        self.dconv_net4 = ResDoubleConv2d(c_1 * 4, c_1 * 8, dropout_net)


class Expansion(nn.Module):
    """
    Expanding path of Unet (Decoder)
    """

    def __init__(self, c_1, dropout_net):
        super(Expansion, self).__init__()
        self.dropout_net = dropout_net
        self.upsample1 = nn.ConvTranspose2d(c_1, c_1 // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(c_1 // 2, c_1 // 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(c_1 // 4, c_1 // 8, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(c_1 // 8, c_1 // 16, kernel_size=2, stride=2)

        self.uconv_net1 = DoubleConv2d(c_1, c_1 // 2, dropout_net)
        self.uconv_net2 = DoubleConv2d(c_1 // 2, c_1 // 4, dropout_net)
        self.uconv_net3 = DoubleConv2d(c_1 // 4, c_1 // 8, dropout_net)
        self.uconv_net4 = DoubleConv2d(c_1 // 8, c_1 // 16, dropout_net)

    def forward(self, x, contract_conv_output):
        dconv1, dconv2, dconv3, dconv4 = contract_conv_output

        x = self.upsample1(x)
        x = self.concatenate(dconv4, x)
        x = self.uconv_net1(x)

        x = self.upsample2(x)
        x = self.concatenate(dconv3, x)
        x = self.uconv_net2(x)

        x = self.upsample3(x)
        x = self.concatenate(dconv2, x)
        x = self.uconv_net3(x)

        x = self.upsample4(x)
        x = self.concatenate(dconv1, x)
        x = self.uconv_net4(x)

        return x

    @staticmethod
    def concatenate(a, b, axis=1, stack=False):
        concat_size = np.min([a.shape[2], b.shape[2]])
        if (a.shape[2] > concat_size):
            a, b = b, a

        midpoint = int(b.shape[2] // 2)  # [-----|-----]
        span = int(concat_size // 2)  # [ <---|---> ] trim the edge of the matrix, reshape for concatenation
        b = b[:, :, midpoint - span:midpoint + span, midpoint - span:midpoint + span]
        res = torch.stack((a, b), axis=axis) if stack else torch.cat((a, b), axis=axis)

        return res


class ResExpansion(Expansion):
    """
    Expanding path of ResUnet (decoder)
    """

    def __init__(self, c_1, dropout_net):
        super(ResExpansion, self).__init__(c_1, dropout_net)
        self.uconv_net1 = ResDoubleConv2d(c_1, c_1 // 2, dropout_net)
        self.uconv_net2 = ResDoubleConv2d(c_1 // 2, c_1 // 4, dropout_net)
        self.uconv_net3 = ResDoubleConv2d(c_1 // 4, c_1 // 8, dropout_net)
        self.uconv_net4 = ResDoubleConv2d(c_1 // 8, c_1 // 16, dropout_net)


class DoubleConv2d(nn.Module):
    """
    Consecutive two convolutional layers
    """

    def __init__(self, in_channel, out_channel, dropout_net):
        super(DoubleConv2d, self).__init__()
        conv_layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            dropout_net,

            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            dropout_net,
        ]
        self.conv_net = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_net(x)


class ResDoubleConv2d(nn.Module):
    """
    Residual block of Consecutive two convolutional layers
    """

    def __init__(self, in_channel, out_channel, dropout_net):
        super(ResDoubleConv2d, self).__init__()
        self.dropout_net = dropout_net
        self.conv_net1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_net2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = F.relu(self.conv_net1(x), inplace=True)
        out = self.conv_net2(out)
        out += self.shortcut(x)
        out = self.dropout_net(out)

        return out
