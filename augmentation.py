import math

import torch
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Function

from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_output = grad_output.contiguous()
        grad_kernel = grad_kernel.contiguous()
        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    if input.device.type == "cpu":
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)


class AdaptiveAugment:
    def __init__(self, ada_aug_target, ada_aug_len, update_every, device):
        self.ada_aug_target = ada_aug_target
        self.ada_aug_len = ada_aug_len
        self.update_every = update_every

        self.ada_update = 0
        self.ada_aug_buf = torch.tensor([0.0, 0.0], device=device)
        self.r_t_stat = 0
        self.ada_aug_p = 0

    @torch.no_grad()
    def tune(self, real_pred):
        self.ada_aug_buf += torch.tensor(
            (torch.sign(real_pred).sum().item(), real_pred.shape[0]),
            device=real_pred.device,
        )
        self.ada_update += 1

        if self.ada_update % self.update_every == 0:
            self.ada_aug_buf = self.ada_aug_buf
            pred_signs, n_pred = self.ada_aug_buf.tolist()

            self.r_t_stat = pred_signs / n_pred

            if self.r_t_stat > self.ada_aug_target:
                sign = 1

            else:
                sign = -1

            self.ada_aug_p += sign * n_pred / self.ada_aug_len
            self.ada_aug_p = min(1, max(0, self.ada_aug_p))
            self.ada_aug_buf.mul_(0)
            self.ada_update = 0

        return self.ada_aug_p


SYM6 = (
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148,
)


def translate_mat(t_x, t_y, device="cpu"):
    batch = t_x.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate

    return mat


def rotate_mat(theta, device="cpu"):
    batch = theta.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batch, 2, 2)
    mat[:, :2, :2] = rot

    return mat


def scale_mat(s_x, s_y, device="cpu"):
    batch = s_x.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y

    return mat


def translate3d_mat(t_x, t_y, t_z):
    batch = t_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y, t_z), 1)
    mat[:, :3, 3] = translate

    return mat


def rotate3d_mat(axis, theta):
    batch = theta.shape[0]

    u_x, u_y, u_z = axis

    eye = torch.eye(3).unsqueeze(0)
    cross = torch.tensor([(0, -u_z, u_y), (u_z, 0, -u_x), (-u_y, u_x, 0)]).unsqueeze(0)
    outer = torch.tensor(axis)
    outer = (outer.unsqueeze(1) * outer).unsqueeze(0)

    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    rot = cos_t * eye + sin_t * cross + (1 - cos_t) * outer

    eye_4 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    eye_4[:, :3, :3] = rot

    return eye_4


def scale3d_mat(s_x, s_y, s_z):
    batch = s_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    mat[:, 2, 2] = s_z

    return mat


def luma_flip_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    flip = 2 * torch.ger(axis, axis) * i.view(-1, 1, 1)

    return eye - flip


def saturation_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    axis = torch.ger(axis, axis)
    saturate = axis + (eye - axis) * i.view(-1, 1, 1)

    return saturate


def lognormal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).log_normal_(mean=mean, std=std)


def category_sample(size, categories, device="cpu"):
    category = torch.tensor(categories, device=device)
    sample = torch.randint(high=len(categories), size=(size,), device=device)

    return category[sample]


def uniform_sample(size, low, high, device="cpu"):
    return torch.empty(size, device=device).uniform_(low, high)


def normal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).normal_(mean, std)


def bernoulli_sample(size, p, device="cpu"):
    return torch.empty(size, device=device).bernoulli_(p)


def random_mat_apply(p, transform, prev, eye, device="cpu"):
    size = transform.shape[0]
    select = bernoulli_sample(size, p, device=device).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye

    return select_transform @ prev


def sample_affine(p, size, height, width, device="cpu"):
    G = torch.eye(3, device=device).unsqueeze(0).repeat(size, 1, 1)
    eye = G

    # flip
    param = category_sample(size, (0, 1))
    Gc = scale_mat(1 - 2.0 * param, torch.ones(size), device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('flip', G, scale_mat(1 - 2.0 * param, torch.ones(size)), sep='\n')

    # 90 rotate
    param = category_sample(size, (0, 3))
    Gc = rotate_mat(-math.pi / 2 * param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('90 rotate', G, rotate_mat(-math.pi / 2 * param), sep='\n')

    # integer translate
    param = uniform_sample((2, size), -0.125, 0.125)
    param_height = torch.round(param[0] * height)
    param_width = torch.round(param[1] * width)
    Gc = translate_mat(param_width, param_height, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('integer translate', G, translate_mat(param_width, param_height), sep='\n')

    # isotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('isotropic scale', G, scale_mat(param, param), sep='\n')

    p_rot = 1 - math.sqrt(1 - p)

    # pre-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param, device=device)
    G = random_mat_apply(p_rot, Gc, G, eye, device=device)
    # print('pre-rotate', G, rotate_mat(-param), sep='\n')

    # anisotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, 1 / param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('anisotropic scale', G, scale_mat(param, 1 / param), sep='\n')

    # post-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param, device=device)
    G = random_mat_apply(p_rot, Gc, G, eye, device=device)
    # print('post-rotate', G, rotate_mat(-param), sep='\n')

    # fractional translate
    param = normal_sample((2, size), std=0.125)
    Gc = translate_mat(param[1] * width, param[0] * height, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('fractional translate', G, translate_mat(param, param), sep='\n')

    return G


def sample_color(p, size):
    C = torch.eye(4).unsqueeze(0).repeat(size, 1, 1)
    eye = C
    axis_val = 1 / math.sqrt(3)
    axis = (axis_val, axis_val, axis_val)

    # brightness
    param = normal_sample(size, std=0.2)
    Cc = translate3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # contrast
    param = lognormal_sample(size, std=0.5 * math.log(2))
    Cc = scale3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # luma flip
    param = category_sample(size, (0, 1))
    Cc = luma_flip_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # hue rotation
    param = uniform_sample(size, -math.pi, math.pi)
    Cc = rotate3d_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # saturation
    param = lognormal_sample(size, std=1 * math.log(2))
    Cc = saturation_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    return C


def make_grid(shape, x0, x1, y0, y1, device):
    n, c, h, w = shape
    grid = torch.empty(n, h, w, 3, device=device)
    grid[:, :, :, 0] = torch.linspace(x0, x1, w, device=device)
    grid[:, :, :, 1] = torch.linspace(y0, y1, h, device=device).unsqueeze(-1)
    grid[:, :, :, 2] = 1

    return grid


def affine_grid(grid, mat):
    n, h, w, _ = grid.shape
    return (grid.view(n, h * w, 3) @ mat.transpose(1, 2)).view(n, h, w, 2)


def get_padding(G, height, width, kernel_size):
    device = G.device

    cx = (width - 1) / 2
    cy = (height - 1) / 2
    cp = torch.tensor(
        [(-cx, -cy, 1), (cx, -cy, 1), (cx, cy, 1), (-cx, cy, 1)], device=device
    )
    cp = G @ cp.T

    pad_k = kernel_size // 4

    pad = cp[:, :2, :].permute(1, 0, 2).flatten(1)
    pad = torch.cat((-pad, pad)).max(1).values
    pad = pad + torch.tensor([pad_k * 2 - cx, pad_k * 2 - cy] * 2, device=device)
    pad = pad.max(torch.tensor([0, 0] * 2, device=device))
    pad = pad.min(torch.tensor([width - 1, height - 1] * 2, device=device))

    pad_x1, pad_y1, pad_x2, pad_y2 = pad.ceil().to(torch.int32)

    return pad_x1, pad_x2, pad_y1, pad_y2


def try_sample_affine_and_pad(img, p, kernel_size, G=None):
    batch, _, height, width = img.shape

    G_try = G

    if G is None:
        G_try = torch.inverse(sample_affine(p, batch, height, width))

    pad_x1, pad_x2, pad_y1, pad_y2 = get_padding(G_try, height, width, kernel_size)

    img_pad = F.pad(img, (pad_x1, pad_x2, pad_y1, pad_y2), mode="reflect")

    return img_pad, G_try, (pad_x1, pad_x2, pad_y1, pad_y2)


class GridSampleForward(autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        out = F.grid_sample(
            input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        ctx.save_for_backward(input, grid)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = GridSampleBackward.apply(grad_output, input, grid)

        return grad_input, grad_grid


class GridSampleBackward(autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")[0]
        grad_input, grad_grid = op(grad_output, input, grid.float(), 0, 0, False, (ctx.needs_input_grad[1], ctx.needs_input_grad[2]))
        ctx.save_for_backward(grid)

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_grid):
        (grid,) = ctx.saved_tensors
        grad_grad_output = None

        if ctx.needs_input_grad[0]:
            grad_grad_output = GridSampleForward.apply(grad_grad_input, grid)

        return grad_grad_output, None, None


grid_sample = GridSampleForward.apply


def scale_mat_single(s_x, s_y):
    return torch.tensor(((s_x, 0, 0), (0, s_y, 0), (0, 0, 1)), dtype=torch.float32)


def translate_mat_single(t_x, t_y):
    return torch.tensor(((1, 0, t_x), (0, 1, t_y), (0, 0, 1)), dtype=torch.float32)


def random_apply_affine(img, p, G=None, antialiasing_kernel=SYM6):
    kernel = antialiasing_kernel
    len_k = len(kernel)

    kernel = torch.as_tensor(kernel).to(img)
    # kernel = torch.ger(kernel, kernel).to(img)
    kernel_flip = torch.flip(kernel, (0,))

    img_pad, G, (pad_x1, pad_x2, pad_y1, pad_y2) = try_sample_affine_and_pad(
        img, p, len_k, G
    )

    G_inv = (
        translate_mat_single((pad_x1 - pad_x2).item() / 2, (pad_y1 - pad_y2).item() / 2)
        @ G
    )
    up_pad = (
        (len_k + 2 - 1) // 2,
        (len_k - 2) // 2,
        (len_k + 2 - 1) // 2,
        (len_k - 2) // 2,
    )
    img_2x = upfirdn2d(img_pad, kernel.unsqueeze(0), up=(2, 1), pad=(*up_pad[:2], 0, 0))
    img_2x = upfirdn2d(img_2x, kernel.unsqueeze(1), up=(1, 2), pad=(0, 0, *up_pad[2:]))
    G_inv = scale_mat_single(2, 2) @ G_inv @ scale_mat_single(1 / 2, 1 / 2)
    G_inv = translate_mat_single(-0.5, -0.5) @ G_inv @ translate_mat_single(0.5, 0.5)
    batch_size, channel, height, width = img.shape
    pad_k = len_k // 4
    shape = (batch_size, channel, (height + pad_k * 2) * 2, (width + pad_k * 2) * 2)
    G_inv = (
        scale_mat_single(2 / img_2x.shape[3], 2 / img_2x.shape[2])
        @ G_inv
        @ scale_mat_single(1 / (2 / shape[3]), 1 / (2 / shape[2]))
    )
    grid = F.affine_grid(G_inv[:, :2, :].to(img_2x), shape, align_corners=False)
    img_affine = grid_sample(img_2x, grid)
    d_p = -pad_k * 2
    down_pad = (
        d_p + (len_k - 2 + 1) // 2,
        d_p + (len_k - 2) // 2,
        d_p + (len_k - 2 + 1) // 2,
        d_p + (len_k - 2) // 2,
    )
    img_down = upfirdn2d(
        img_affine, kernel_flip.unsqueeze(0), down=(2, 1), pad=(*down_pad[:2], 0, 0)
    )
    img_down = upfirdn2d(
        img_down, kernel_flip.unsqueeze(1), down=(1, 2), pad=(0, 0, *down_pad[2:])
    )

    return img_down, G


def apply_color(img, mat):
    batch = img.shape[0]
    img = img.permute(0, 2, 3, 1)
    mat_mul = mat[:, :3, :3].transpose(1, 2).view(batch, 1, 3, 3)
    mat_add = mat[:, :3, 3].view(batch, 1, 1, 3)
    img = img @ mat_mul + mat_add
    img = img.permute(0, 3, 1, 2)

    return img


def random_apply_color(img, p, C=None):
    if C is None:
        C = sample_color(p, img.shape[0])

    img = apply_color(img, C.to(img))

    return img, C


def augment(img, p, transform_matrix=(None, None)):
    img, G = random_apply_affine(img, p, transform_matrix[0])
    img, C = random_apply_color(img, p, transform_matrix[1])

    return img, (G, C)
