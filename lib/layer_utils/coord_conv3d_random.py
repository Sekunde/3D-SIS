import torch
from torch import nn


class AddCoordinates3d(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf), but adapted to work with 3D
    convolutions.

    This module concatenates coordinate information (`x`, `y`, `z` and `r`) with
    given input tensor.

    `x`, `y` and `z` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input grid. Default: `False`

    Shape:
        - Input: `(N, C_{in}, dimX_{in}, dimY_{in}, dimZ_{in})`
        - Output: `(N, (C_{in} + 3) or (C_{in} + 4), dimX_{in}, dimY_{in}, dimZ_{in})`

    Examples:
        >>> coord_adder = AddCoordinates3d(True)
        >>> input = torch.randn(8, 3, 8, 8, 8)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates3d(True)
        >>> input = torch.randn(8, 3, 8, 8, 8).cuda()
        >>> output = coord_adder(input)

        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 8, 8, 8).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, window_size=16, step=1, with_r=False):
        self.with_r = with_r
        self.window_size = window_size
        self.step = step

    def __normalize__(self, x):
        return 2 * (x % self.window_size) / self.window_size - 1


    def __call__(self, grid, random_shift):
        print('used')
        batch_size, _, grid_dimX, grid_dimY, grid_dimZ = grid.size()
        k = 1.0

        x_coords = self.__normalize__((k * torch.arange(0, grid_dimX*self.step, step=self.step, dtype=torch.float32).unsqueeze(1).unsqueeze(1
            )+random_shift[0]).expand(grid_dimX, grid_dimY, grid_dimZ))
        y_coords = self.__normalize__((k * torch.arange(0, grid_dimY*self.step, step=self.step, dtype=torch.float32).unsqueeze(1).unsqueeze(0
            )+random_shift[1]).expand(grid_dimX, grid_dimY, grid_dimZ))
        z_coords = self.__normalize__((k * torch.arange(0, grid_dimZ*self.step, step=self.step, dtype=torch.float32).unsqueeze(0).unsqueeze(0
            )+random_shift[2]).expand(grid_dimX, grid_dimY, grid_dimZ))

        coords = torch.stack((x_coords, y_coords, z_coords), dim=0)

        if self.with_r:
            rs = ((x_coords ** 2) + (y_coords ** 2) + (z_coords ** 2)) ** 0.5
            rs = k * rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1, 1)

        grid = torch.cat((coords.to(grid.device), grid), dim=1)
        return grid


class CoordConv3d(nn.Module):

    r"""3D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf), but adapted to 
    work with 3D convolutions.

    Args:
        Same as `torch.nn.Conv3d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input grid. Default: `False`

    Shape:
        - Input: `(N, C_{in}, dimX_{in}, dimY_{in}, dimZ_{in})`
        - Output: `(N, C_{out}, dimX_{out}, dimY_{out}, dimZ_{out})`

    Examples:
        >>> coord_conv = CoordConv3d(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 8, 8, 8)
        >>> output = coord_conv(input)

        >>> coord_conv = CoordConv3d(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 8, 8, 8).cuda()
        >>> output = coord_conv(input)

        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv3d(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 8, 8, 8).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False, step=1, zeroWeightInitialization=True):
        super(CoordConv3d, self).__init__()

        coords_channels = 3
        if with_r:
            coords_channels += 1
        
        in_channels += coords_channels

        self.conv_layer = nn.Conv3d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_adder = AddCoordinates3d(16, step, with_r)

        if zeroWeightInitialization:
            self._initialize_weights(coords_channels)

    def _initialize_weights(self, coords_channels):
        weights = self.conv_layer.weight.data

        # We keep the default initialization of almost all Conv3d weights, we only set
        # weights related to coordinate inputs to zero.
        out_channels, in_channels, k_dimX, k_dimY, k_dimZ = weights.size()
        assert(coords_channels < in_channels)

        weights[:, 0:coords_channels, :, :, :] = torch.zeros(out_channels, coords_channels, k_dimX, k_dimY, k_dimZ).to(weights.device)

        self.conv_layer.weight = nn.Parameter(weights)

    def forward(self, x, random_shift):
        x = self.coord_adder(x, random_shift)
        x = self.conv_layer(x)

        return x


class CoordConvTranspose3d(nn.Module):

    r"""3D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf), but adapted to 
    work with 3D convolutions.

    Args:
        Same as `torch.nn.ConvTranspose3d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input grid. Default: `False`

    Shape:
        - Input: `(N, C_{in}, dimX_{in}, dimY_{in}, dimZ_{in})`
        - Output: `(N, C_{out}, dimX_{out}, dimY_{out}, dimZ_{out})`

    Examples:
        >>> coord_conv_tr = CoordConvTranspose3d(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 8, 8, 8)
        >>> output = coord_conv_tr(input)

        >>> coord_conv_tr = CoordConvTranspose3d(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 8, 8, 8).cuda()
        >>> output = coord_conv_tr(input)

        >>> device = torch.device("cuda:0")
        >>> coord_conv_tr = CoordConvTranspose3d(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 8, 8, 8).to(device)
        >>> output = coord_conv_tr(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, with_r=False):
        super(CoordConvTranspose3d, self).__init__()

        in_channels += 3
        if with_r:
            in_channels += 1

        self.conv_tr_layer = nn.ConvTranspose3d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)

        self.coord_adder = AddCoordinates3d(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_tr_layer(x)
        return x


class CoordConvNet3d(nn.Module):

    r"""Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf), but adapted to work with 
    3D convolutions.

    This module adds coordinate information to inputs of each 3D convolution
    module (`torch.nn.Conv3d`).

    Assumption: ConvNet Model must contain single `Sequential` container
    (`torch.nn.modules.container.Sequential`).

    Args:
        cnn_model: A ConvNet model that must contain single `Sequential`
            container (`torch.nn.modules.container.Sequential`).
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input grid. Default: `False`

    Shape:
        - Input: Same as the input of the model.
        - Output: A list that contains all outputs (including
            intermediate outputs) of the model.

    Examples:
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet3d(cnn_model, True)
        >>> input = torch.randn(8, 3, 8, 8, 8)
        >>> outputs = cnn_model(input)

        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet3d(cnn_model, True).cuda()
        >>> input = torch.randn(8, 3, 8, 8, 8).cuda()
        >>> outputs = cnn_model(input)

        >>> device = torch.device("cuda:0")
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet3d(cnn_model, True).to(device)
        >>> input = torch.randn(8, 3, 8, 8, 8).to(device)
        >>> outputs = cnn_model(input)
    """

    def __init__(self, cnn_model, with_r=False):
        super(CoordConvNet3d, self).__init__()

        self.with_r = with_r

        self.cnn_model = cnn_model
        self.__get_model()
        self.__update_weights()

        self.coord_adder = AddCoordinates3d(self.with_r)

    def __get_model(self):
        for module in list(self.cnn_model.modules()):
            if module.__class__ == torch.nn.modules.container.Sequential:
                self.cnn_model = module
                break

    def __update_weights(self):
        coord_channels = 3
        if self.with_r:
            coord_channels += 1

        for l in list(self.cnn_model.modules()):
            if l.__str__().startswith('Conv3d'):
                weights = l.weight.data

                out_channels, in_channels, k_dimX, k_dimY, k_dimZ = weights.size()

                coord_weights = torch.zeros(out_channels, coord_channels,
                                            k_dimX, k_dimY, k_dimZ)

                weights = torch.cat((coord_weights.to(weights.device),
                                     weights), dim=1)
                weights = nn.Parameter(weights)

                l.weight = weights
                l.in_channels += coord_channels

    def __get_outputs(self, x):
        outputs = []
        for layer_name, layer in self.cnn_model._modules.items():
            if layer.__str__().startswith('Conv3d'):
                x = self.coord_adder(x)
            x = layer(x)
            outputs.append(x)

        return outputs

    def forward(self, x):
        return self.__get_outputs(x)
