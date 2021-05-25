import torch
from torch.autograd.function import Function


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
            shape_mul = torch.where(shape.abs() < 10e-7,
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
            shape_mul = torch.where(shape.abs() < 10e-20,
                                    torch.tensor([1.], dtype=torch.double),
                                    1 / (48**(1/3)) *
                                    (section / shape)**(2/3)
                                    )
            shape_grad = grad_output * shape_mul
        if ctx.needs_input_grad[1]:
            section_mul = torch.where(section.abs() < 10e-20,
                                      torch.tensor([1.], dtype=torch.double),
                                      1 / (6**(1/3)) *
                                      (shape / section)**(1/3)
                                      )
            section_grad = grad_output * section_mul

        return shape_grad, section_grad
