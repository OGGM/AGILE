import os.path

import torch
import numpy as np
import pandas as pd
import pytest
from torch.autograd import gradcheck
from combine1d.core.special_gradient_functions import para_width_from_thick, \
    para_thick_from_section, SolveBandedPyTorch
from combine1d.core.torch_interp1d import Interp1d


pytestmark = pytest.mark.filterwarnings("ignore:<class 'combine1d.core.torch_interp1d.Interp1d'> "
                                        "should not be instantiated.:DeprecationWarning")


def test_parabolic_functions():
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

    # check if it also works at the given points
    input_parameters = (x, y, x)
    assert gradcheck(Interp1d(), input_parameters)


def test_SolveBandedPyTorch(data_dir):
    # just creating some dummy data for testing
    Amat_banded = torch.tensor([[0., .2, .2, .2],
                                [1., 1., 1., 1.],
                                [.1, .1, .1, 0.],
                                ],
                               dtype=torch.double,
                               requires_grad=True)
    rhs = torch.tensor([1., 2., 3., 4.],
                       dtype=torch.double,
                       requires_grad=True)
    assert gradcheck(SolveBandedPyTorch.apply, (Amat_banded, rhs))

    # with more realistic data
    Amat_banded = torch.tensor(
        np.array(pd.read_csv(
            os.path.join(data_dir, 'Amat_banded_test.csv'),
            header=None)),
        dtype=torch.double,
        requires_grad=True)
    rhs = torch.tensor(
        np.squeeze(np.array(
            pd.read_csv(os.path.join(data_dir, 'rhs_test.csv'),
                        header=None))),
        dtype=torch.double,
        requires_grad=True)
    assert gradcheck(SolveBandedPyTorch.apply, (Amat_banded, rhs))
