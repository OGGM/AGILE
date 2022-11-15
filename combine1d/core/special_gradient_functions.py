import torch
from torch.autograd.function import Function
from scipy.linalg import solve_banded
import numpy as np


class para_width_from_thick(Function):
    @staticmethod
    def forward(ctx, shape, thick):
        torch_type = thick.dtype
        device = thick.device

        result = torch.sqrt(torch.tensor(4.,
                                         dtype=torch_type,
                                         device=device,
                                         requires_grad=False) * thick / shape)

        # save parameters for gradient calculation
        ctx.save_for_backward(thick, shape)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # get parameters from forward run
        thick, shape = ctx.saved_tensors

        # only calculate gradients when needed
        shape_grad = thick_grad = None
        if ctx.needs_input_grad[0]:
            shape_mul = torch.where(shape.abs() < 10e-7,
                                    torch.tensor([1.],
                                                 dtype=shape.dtype,
                                                 device=shape.device),
                                    torch.sqrt(thick / shape.pow(3)) *
                                    torch.tensor(- 1,
                                                 dtype=shape.dtype,
                                                 device=shape.device)
                                    )
            shape_grad = grad_output * shape_mul
        if ctx.needs_input_grad[1]:
            thick_mul = torch.where((thick.abs() < 10e-10) |
                                    (shape.abs() < 10e-10),
                                    torch.tensor([1.],
                                                 dtype=thick.dtype,
                                                 device=thick.device),
                                    torch.tensor(1.,
                                                 dtype=thick.dtype,
                                                 device=thick.device) / (torch.sqrt(thick * shape))
                                    )
            thick_grad = grad_output * thick_mul

        return shape_grad, thick_grad


class para_thick_from_section(Function):
    @staticmethod
    def forward(ctx, shape, section):
        torch_type = section.dtype
        device = section.device

        result = (torch.tensor(0.75,
                               dtype=torch_type,
                               device=device) * section * torch.sqrt(shape)).pow(2. / 3.)

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
            shape_mul = torch.where(shape.abs() < 10e-20,
                                    torch.tensor([1.], dtype=shape.dtype, device=shape.device),
                                    torch.tensor(1.,
                                                 dtype=shape.dtype,
                                                 device=shape.device) /
                                    (torch.tensor(48,
                                                  dtype=shape.dtype,
                                                  device=shape.device).pow(1. / 3.)) *
                                    (section / shape).pow(2. / 3.)
                                    )
            shape_grad = grad_output * shape_mul
        if ctx.needs_input_grad[1]:
            section_mul = torch.where(section.abs() < 10e-20,
                                      torch.tensor([1.],
                                                   dtype=section.dtype, device=section.device),
                                      torch.tensor(1.,
                                                   dtype=section.dtype,
                                                   device=section.device) /
                                      (torch.tensor(6.,
                                                    dtype=section.dtype,
                                                    device=section.device).pow(1. / 3.)) *
                                      (shape / section).pow(1. / 3.)
                                      )
            section_grad = grad_output * section_mul

        return shape_grad, section_grad


class SolveBandedPyTorch(Function):
    """This is a pytorch version for the scipy function solve_banded. It also
    calculates the gradients icorporating the adjoint from the eq. (7) and (8)
    from Goldberg et al. 2013.
    Explanation of solve_banded can be found here
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
    """

    @staticmethod
    def forward(ctx, Amat_banded, rhs):
        # solve for the new thickness using scipy and convert to tensor
        h_new_np = solve_banded((1, 1), Amat_banded, rhs)

        # Saving inputs and outputs for backward use `save_for_backward`
        ctx.save_for_backward(Amat_banded)

        # Save non-tensors and non-inputs/non-outputs directly on ctx
        ctx.h_new_np = h_new_np

        # convert resulting new thickness to PyTorch tensor and return
        return torch.tensor(h_new_np,
                            dtype=Amat_banded.dtype,
                            device=Amat_banded.device,
                            requires_grad=True)

    @staticmethod
    def backward(ctx, grad_out):

        # this manual gradient implements (7), (8) of
        # Goldberg, D. N. and Heimbach, P.: Parameter and state estimation
        # with a time-dependent adjoint marine ice sheet model, in
        # The Cryosphere, 7, 1659–1678

        # X = A^{-1} B

        # we must find the gradient of 2 objects, A and B

        # 1. \delta^* B = A^{-T} \delta^* X (where \delta^* X = grad_out)
        # 2. \delta^* A = - (\delta^* B) (X^T) <-- outer tensor product;
        #                                          but ZERO outside of sparsity
        #                                          pattern

        # get the saved values
        Amat_banded, = ctx.saved_tensors
        h_new_np = ctx.h_new_np

        # get transpose of Amat_banded, make use of tridiagonal structure
        Amat_banded_np = Amat_banded.cpu().detach().numpy()
        Amat_banded_transpose = Amat_banded_np.copy()
        Amat_banded_transpose[0, 1:] = Amat_banded_np[2, :-1]
        Amat_banded_transpose[2, :-1] = Amat_banded_np[0, 1:]

        # only calculate gradients if needed, otherwise they should be None
        # but in this case it is assumed that the gradient is always caluclated
        # for both, just let the code here for documentation reasons
        # Amat_banded_grad = rhs_grad = None
        # if ctx.needs_input_grad[1]: # for rhs_grad
        # # if ctx.needs_input_grad[0]:  # for Amat_banded_grad

        # calculate gradient of rhs
        rhs_grad_np = solve_banded((1, 1), Amat_banded_transpose, grad_out)

        # calculate Amat_banded_grad, it is derived from a tensordot product,
        # but only the three diagonals of the banded matrix are needed in the
        # end, so here only calculate the point wise product for the three
        # needed diagonals
        # original code for clarification:
        # amat_gradient = -1. * np.tensordot(rhs_grad,h_new,axes=0)
        # # amat_gradient is a dense matrix -- but we need only recover
        # # those which are nonzero in the sparse structure (the tridiagonal
        # # entries)
        # Amat_banded_grad = torch.zeros(Amat_banded.size())
        # Amat_banded_grad[0,1:] = torch.tensor(np.diag(amat_gradient,1))
        # Amat_banded_grad[1,:] = torch.tensor(np.diag(amat_gradient))
        # Amat_banded_grad[2,:-1] = torch.tensor(np.diag(amat_gradient,-1))
        Amat_banded_grad_np = np.zeros(Amat_banded_np.shape)
        Amat_banded_grad_np[0, 1:] = -1. * rhs_grad_np[:-1] * h_new_np[1:]
        Amat_banded_grad_np[1, :] = -1. * rhs_grad_np * h_new_np
        Amat_banded_grad_np[2, :-1] = -1. * rhs_grad_np[1:] * h_new_np[:-1]

        # convert to rhs_grad and Amat_banded_grad to tensor
        rhs_grad = torch.tensor(rhs_grad_np,
                                dtype=Amat_banded.dtype,
                                device=Amat_banded.device,
                                requires_grad=True
                                )
        Amat_banded_grad = torch.tensor(Amat_banded_grad_np,
                                        dtype=Amat_banded.dtype,
                                        device=Amat_banded.device,
                                        requires_grad=True)

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return Amat_banded_grad, rhs_grad
