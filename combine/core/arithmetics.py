import numpy as np
from scipy.signal import convolve2d
import torch
from torch.autograd.function import Function


def mean_BIAS(a1, a2, ice_mask=None):
    if ice_mask is None:
        dev = a1 - a2
    else:
        dev = np.ma.masked_array(a1 - a2, mask=np.logical_not(ice_mask))
    return dev.mean()


def RMSE(a1, a2, ice_mask=None):
    if ice_mask is None:
        dev = a1 - a2
    else:
        dev = np.ma.masked_array(a1 - a2, mask=np.logical_not(ice_mask))
    return np.sqrt((dev**2).mean())


def max_dif(a1, a2):
    return np.max(np.abs(a1 - a2))


def percentiles(a1, a2, ice_mask, q=[5, 25, 50, 75, 95]):
    if ice_mask is None:
        dev = a1 - a2
    else:
        dev = (a1 - a2)[ice_mask]
    return np.percentile(dev, q)


def compute_inner_mask(ice_mask, full_array=False):
    conv_array = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # TODO
    if full_array:
        conv_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # TODO
    inner_mask = convolve2d(ice_mask, conv_array, mode='same') == \
                 conv_array.sum()
    return inner_mask


def to_torch_tensor(val, torch_type, requires_grad=False):
    if type(val) == np.ndarray:
        val = torch.from_numpy(val).to(torch_type)
    elif val is None:
        pass
    elif type(val) != torch.Tensor:
        val = torch.tensor(val,
                           dtype=torch_type,
                           requires_grad=requires_grad)

    return val


def to_numpy_array(val):
    if type(val) == torch.Tensor:
        val = val.detach().numpy()
    elif val is None:
        pass
    elif type(val) != np.ndarray:
        val = np.array(val)

    return val


def magnitude(x):
    x = np.where(x != 0,
                 x, 1e-100)
    return np.floor(np.log10(np.abs(x))).astype(float)


class para_width_from_thick(Function):
    @staticmethod
    def forward(ctx, shape, thick):
        result = torch.sqrt(4. * thick / shape)

        # save parameters for gradient calculation
        ctx.save_for_backward(thick, shape)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # get parameters from forward run
        thick, shape, = ctx.saved_tensors

        # only calculate gradients when needed
        shape_grad = thick_grad = None
        if ctx.needs_input_grad[0]:
            shape_mul = torch.where(shape.abs() < 10e-10,
                                    torch.tensor([1.],
                                                 dtype=torch.double),
                                    torch.sqrt(thick / shape.pow(3)) * (- 1)
                                    )
            shape_grad = grad_output * shape_mul
        if ctx.needs_input_grad[1]:
            thick_mul = torch.where((thick.abs() < 10e-10) |
                                    (shape.abs() < 10e-10),
                                    torch.tensor([1.],
                                                 dtype=torch.double),
                                    1 / (torch.sqrt(thick * shape))
                                    )
            thick_grad = grad_output * thick_mul

        return shape_grad, thick_grad


class para_thick_from_section(Function):
    @staticmethod
    def forward(ctx, shape, section):
        result = (0.75 * section * torch.sqrt(shape))**(2./3.)

        # save parameters for gradient calculation
        ctx.save_for_backward(section, shape)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # get parameters from forward run
        section, shape = ctx.saved_tensors

        # only calculate gradients when needed
        shape_grad = section_grad = None
        if ctx.needs_input_grad[0]:
            shape_mul = torch.where(shape.abs() < 10e-10,
                                    torch.tensor([1.], dtype=torch.double),
                                    1 / (2 * 6**(1/3)) *
                                    (section / shape)**(2/3)
                                    )
            shape_grad = grad_output * shape_mul
        if ctx.needs_input_grad[1]:
            section_mul = torch.where(section.abs() < 10e-10,
                                      torch.tensor([1.], dtype=torch.double),
                                      1 / (6**(1/3)) *
                                      (shape / section)**(1/3)
                                      )
            section_grad = grad_output * section_mul

        return shape_grad, section_grad
