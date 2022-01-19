import torch
from torch.autograd import gradcheck
from combine1d.core.special_gradient_functions import para_width_from_thick,\
    para_thick_from_section
from combine1d.core.torch_interp1d import Interp1d


def test_special_gradient_functions():
    test_shape = torch.abs(torch.randn(1000, dtype=torch.double, requires_grad=True))
    test_thick = torch.abs(torch.randn(1000, dtype=torch.double, requires_grad=True)) * 1000

    input_shape_thick = (test_shape, test_thick)
    assert gradcheck(para_width_from_thick.apply, input_shape_thick)

    test_section = torch.abs(torch.randn(1000, dtype=torch.double, requires_grad=True)) * 100

    input_shape_section = (test_shape, test_section)
    assert gradcheck(para_thick_from_section.apply, input_shape_section)

def test_Interp1d_gradient_calculation():
    x = torch.arange(0, 10, step=1, dtype=torch.double, requires_grad=True)
    y = torch.arange(0, 10, step=1, dtype=torch.double, requires_grad=True).pow(2)

    x_new = torch.arange(0.5, 9.5, step=1, dtype=torch.double, requires_grad=True)

    input_parameters = (x, y, x_new)
    assert gradcheck(Interp1d(), input_parameters)
