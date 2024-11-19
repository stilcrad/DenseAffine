import torch.nn as nn


def make_layers(cfg, batch_norm=True, extra_in_channels=0,
                nonlinearity=None, nonlinearity_kwargs=None, co_ord_conv=False):
    nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)
    layers = []
    in_channels = cfg[0] + extra_in_channels
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # elif v == 'U':
        # layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
        elif v == 'U':
            # in_channels_new = round(in_channels / 2)
            in_channels_new = in_channels
            layers += [nn.ConvTranspose2d(in_channels, in_channels_new, 4, 2, 1, bias=False)]
            if batch_norm:
                layers += [nn.BatchNorm2d(in_channels_new)]
            layers += [nonlinearity]
            in_channels = in_channels_new
        elif v == 'L':
            pass
            # layers += [knn.ActivationMap()]
        else:
            layers += [nn.ReplicationPad2d(1)]
            if co_ord_conv:
                pass
                # layers += [knn.Coords()]
            layers += [nn.Conv2d(in_channels + 2 * co_ord_conv, v, kernel_size=3)]
            if batch_norm:
                layers += [nn.BatchNorm2d(v)]
            layers += [nonlinearity]

            in_channels = v
    return nn.Sequential(*layers)


"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
U -> Bilinear upsample
"""

decoder_cfg = {
    # 'A': [512, 'U', 512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 'U', 64],
    'A': [512, 'U', 512, 512, 'U', 256, 256, 'U', 128, 128, 'U', 64, 64, 'U'],
    # 'A': [512, 'U', 'U', 'U', 'U', 'U'],
    'F': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 64],
    'VGG_PONG': [32, 'U', 16, 'U', 16],
    'VGG_PONG_TRIVIAL': [16, 16],
    'VGG_PONG_LAYERNECK': [32, 32, 16, 16],
    'VGG_PACMAN': [16, 32, 32, 16],
    'VGG_PACMAN_2': [64, 'U', 32, 32, 16],
}

vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG_PONG': [16, 'M', 16, 'M', 32],
    'VGG_PONG_TRIVIAL': [16, 16],
    'VGG_PONG_LAYERNECK': [16, 32],
    'VGG_PACMAN': [16, 32, 32, 16],
    'VGG_PACMAN_2': [16, 32, 32, 'M', 64],
    'MAPPER': [8, 8],
}
