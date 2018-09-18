from cobbi.utils import test_cases
from cobbi.inversion import *
import matplotlib.pyplot as plt
#from torch.optim import LBFGS

case = test_cases.mayan_ice_cap
case = test_cases.mayan_ice_cap
case = test_cases.blafell
case.dx = 600
case.smooth_border_px = 5

y0 = 0
y_spinup_end = 2000
y_end = 2700
lamb1 = 0.
lamb2 = 0.
lamb3 = 0.
lamb4 = 2.
max_iter = 10

start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)
cost_func = create_cost_function_true_surf(start_surf, reference_surf,
                                           ice_mask, case.dx, mb,
                                           y_spinup_end, y_end,
                                           lamb1=lamb1, lamb2=lamb2,
                                           lamb3=lamb3, lamb4=lamb4,
                                           return_calculated_surface=True)

bed_0 = get_first_guess(reference_surf, ice_mask, case.dx)

i = 0
cost = 1e6
bed = bed_0.detach().numpy()
k = 0.5
max_abs_bed_update = 30  # only allow 30m bed update per iteration


plt.ion()
plt.figure();
plt.imshow(reference_surf);
plt.title('True surface')
plt.show()
plt.figure();
plt.imshow(bed_2d);
plt.title('True bed')
plt.show()

#Simple fixpoint iteration
while i < max_iter and cost > 1e3:
    cost, grad, surf = cost_func(bed)

    #plt.figure()
    #plt.imshow(surf)
    #plt.title('Modelled surface {:d}'.format(i))
    #plt.show()

    #plt.figure()
    #plt.imshow(bed)
    #plt.title('Modelled bed {:d}'.format(i))
    #plt.show()

    plt.figure()
    plt.imshow(surf - reference_surf)
    plt.title('Surface difference {:d}'.format(i))
    plt.show()

    plt.figure()
    plt.imshow(grad.reshape(bed.shape))
    plt.title('Gradient {:d}'.format(i))
    plt.show()

    plt.figure()
    plt.imshow(bed - bed_2d)
    plt.title('Bed difference {:d}'.format(i))
    plt.show()

    print('-----------------------------------------')
    print('Iteration: {:d}'.format(i))
    print('Cost: {:g}'.format(cost))
    print('Bed RMSE: ', RMSE(bed, bed_2d.numpy()))
    print('Bed Max_diff: ', np.max(np.abs(bed - bed_2d.numpy())))
    print('Surface RMSE: ', RMSE(surf, reference_surf.detach().numpy()))
    print('Surface Max_diff: ',
          np.max(np.abs(surf - reference_surf.detach().numpy())))

    cut_grad = grad.reshape(bed.shape) * ice_mask.detach().numpy()

    k = cost / np.abs(cut_grad).sum() * 0.1  # should
    # theoretically
    # compensate 10%
    # of cost
    bed_update = np.where(np.abs(cut_grad) > 0., cut_grad, 0)
    bed_update = -k * bed_update
    bed_update = bed_update.clip(min=-max_abs_bed_update,
                                 max=max_abs_bed_update)
    bed = bed + bed_update

    plt.figure()
    plt.imshow(bed_update)
    plt.title('Bed update {:d}'.format(i))
    plt.show()

    i = i + 1



print('end')