import torch
import matplotlib.pyplot as plt


class LocalMeanSquaredDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, modelled_surf, observed_surf, bed):
        ctx.save_for_backward(modelled_surf, observed_surf, bed)
        msd = (modelled_surf - observed_surf).pow(2).mean()
        return msd

    @staticmethod
    def backward(ctx, grad_output):
        modelled_surf, observed_surf, bed = ctx.saved_tensors
        grad_modelled_surf = (modelled_surf - observed_surf) * -1.
        return None, None, grad_modelled_surf

a = torch.zeros((7, 7), requires_grad=False)
a[2:-2, 2:-2] = 1
a.requires_grad = True

b = torch.zeros(a.shape)
b[:-1, :-1] = a[1:, 1:] - 5 * a[:-1, :-1]

observed = torch.ones(b.shape, requires_grad=False)

lmsd = LocalMeanSquaredDifference.apply

loss = lmsd(b, observed, a)
#loss = (b.detach().numpy())
loss.backward()

#with torch.no_grad():
grad = a.grad
print(grad)
#a.grad.zero_()

print('end')

