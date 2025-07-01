import torch
from torch import nn
import torch.nn.functional as F


def ramp_filter(size):
    image_n = torch.cat([
        torch.arange(1, size / 2 + 1, 2, dtype=torch.int),
        torch.arange(size / 2 - 1, 0, -2, dtype=torch.int),
    ])

    image_filter = torch.zeros(size, dtype=torch.double)
    image_filter[0] = 0.25
    image_filter[1::2] = -1 / (PI * image_n) ** 2

    fourier_filter = torch.rfft(image_filter, 1, onesided=False)
    fourier_filter[:, 1] = fourier_filter[:, 0]

    return 2*fourier_filter

class AbstractFilter(nn.Module):
    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = \
            max(64, int(2**(2*torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0, 0, 0, pad_width))
        fourier_filter = ramp_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(fourier_filter)
        fourier_filter = fourier_filter.unsqueeze(-2)
        projection = rfft(padded_tensor, axis=2)*fourier_filter
        return irfft(projection, axis=2)[:, :, :input_size, :].to(x.dtype)

    def create_filter(self, fourier_ramp):
        raise NotImplementedError

class RampFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        return fourier_ramp

class HannFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        n = torch.arange(0, fourier_ramp.shape[0])
        hann = (0.5 - 0.5*(2.0*PI*n/(fourier_ramp.shape[0]-1)).cos()).to(fourier_ramp.device)
        return fourier_ramp*hann.roll(hann.shape[0]//2, 0).unsqueeze(-1)

class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(ramp_filter(filter_size)[..., 0].view(-1, 1))

    def forward(self, x):
        fourier_filter = self.filter.unsqueeze(-1).repeat(1, 1, 2).to(x.device)
        projection = rfft(x, axis=2) * fourier_filter
        return irfft(projection, axis=2).to(x.dtype)

    def create_filter(self, fourier_ramp):
        raise NotImplementedError



import torch
from torch import nn
import torch.nn.functional as F


class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape

        # 初始化out张量，使用动态的宽度尺寸
        max_width = W
        out = torch.zeros(N, C, max_width, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            rotated_sum = rotated.sum(2)
            
            # 如果尺寸不匹配，则调整 rotated_sum 的尺寸
            if rotated_sum.shape[2] != max_width:
                rotated_sum = F.interpolate(rotated_sum.unsqueeze(-1), size=(max_width, 1), mode='nearest').squeeze(-1)
            
            out[..., i] = rotated_sum

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids

class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.deg2rad = lambda x: deg2rad(x, dtype)
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size/SQRT2).floor()) if not self.circle else it_size
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        x = self.filter(x)

        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco *= PI.to(reco.device)/(2*len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size)//2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2*in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _xy_to_t(self, theta):
        return self.xgrid*self.deg2rad(theta).cos() - self.ygrid*self.deg2rad(theta).sin()

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        all_grids = []
        for i_theta, theta in enumerate(angles):
            X = torch.ones([grid_size]*2, dtype=self.dtype)*i_theta*2./(len(angles)-1)-1.
            Y = self._xy_to_t(theta)
            all_grids.append(torch.stack((X, Y), dim=-1).unsqueeze(0))
        return all_grids


import torch
from torch import nn


class Stackgram(nn.Module):
    def __init__(self, out_size, theta=None, circle=True, mode='nearest', dtype=torch.float):
        super(Stackgram, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size = out_size if circle else int((SQRT2*out_size).ceil())
        self.dtype = dtype
        self.all_grids = self._create_grids(self.theta, in_size)
        self.mode = mode

    def forward(self, x):
        stackgram = torch.zeros(x.shape[0], len(self.theta), self.in_size, self.in_size, device=x.device, dtype=self.dtype)

        for i_theta in range(len(self.theta)):
            repline = x[..., i_theta]
            repline = repline.unsqueeze(-1).repeat(1,1,1,repline.shape[2])
            linogram = grid_sample(repline, self.all_grids[i_theta].repeat(x.shape[0], 1, 1, 1).to(x.device), mode=self.mode)
            stackgram[:, i_theta] = linogram

        return stackgram

    def _create_grids(self, angles, grid_size):
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            t = deg2rad(theta)
            R = torch.tensor([
                [t.sin(), t.cos(), 0.],
                [t.cos(), -t.sin(), 0.],
            ], dtype=self.dtype).unsqueeze(0)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids

class IStackgram(nn.Module):
    def __init__(self, out_size, theta=None, circle=True, mode='bilinear', dtype=torch.float):
        super(IStackgram, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size = out_size if circle else int((SQRT2*out_size).ceil())
        self.dtype = dtype
        self.all_grids = self._create_grids(self.theta, in_size)
        self.mode = mode

    def forward(self, x):
        sinogram = torch.zeros(x.shape[0], 1, self.in_size, len(self.theta), device=x.device, dtype=self.dtype)

        for i_theta in range(len(self.theta)):
            linogram = x[:, i_theta].unsqueeze(1)
            repline = grid_sample(linogram, self.all_grids[i_theta].repeat(x.shape[0], 1, 1, 1).to(x.device), mode=self.mode)
            repline = repline[..., repline.shape[-1]//2]
            sinogram[..., i_theta] = repline

        return sinogram

    def _create_grids(self, angles, grid_size):
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            t = deg2rad(theta)
            R = torch.tensor([
                [t.sin(), t.cos(), 0.],
                [t.cos(), -t.sin(), 0.],
            ], dtype=self.dtype).unsqueeze(0)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids



import torch
import torch.nn.functional as F

if torch.__version__ > '1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1, dtype=torch.double).atan()
SQRT2 = (2*torch.ones(1, dtype=torch.double)).sqrt()

def deg2rad(x, dtype=torch.float):
    return (x*PI/180).to(dtype)

def rfft(tensor, axis=-1):
    ndim = tensor.ndim
    if axis < 0:
        axis %= ndim
    tensor = tensor.transpose(axis, ndim-1)
    fft_tensor = torch.rfft(
        tensor,
        1,
        normalized=False,
        onesided=False,
    )
    return fft_tensor.transpose(axis, ndim-1)

def irfft(tensor, axis):
    assert 0 <= axis < tensor.ndim
    tensor = tensor.transpose(axis, tensor.ndim-2)
    ifft_tensor = torch.ifft(
        tensor,
        1,
        normalized=False,
    )[..., 0]
    return ifft_tensor.transpose(axis, tensor.ndim-2)