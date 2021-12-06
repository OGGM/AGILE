import torch
from torch.autograd.function import Function


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
