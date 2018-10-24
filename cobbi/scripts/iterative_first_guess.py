import torch
import numpy as np
import matplotlib.pyplot as plt
from cobbi.utils import test_cases
from cobbi.inversion import *
import matplotlib.colors as colors
import os

case = test_cases.Giluwe
#case.dx = 600
#case.smooth_border_px = 2
#print(case.mb_grad)
#print(case.ela_h)

basedir = '/data/philipp/tests/cases/'
basedir = os.path.join(basedir, case.name)
if not os.path.exists(basedir):
    os.makedirs(basedir, exist_ok=True)

y0 = 0
y_spinup_end = 2000
y_end = 2800

start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

start_surf = start_surf.detach().numpy()
reference_surf = reference_surf.detach().numpy()
ice_mask = ice_mask.detach().numpy()
bed_2d = bed_2d.detach().numpy()

#first_bed_guess = first_guess(reference_surf, ice_mask, case.dx)

def run_forward(mb_model, b, spinup_surf):
    with torch.no_grad():
        start_surf = torch.tensor(spinup_surf, dtype=torch.float,
                                  requires_grad=False)
        bed = torch.tensor(b.reshape(start_surf.shape), dtype=torch.float,
                           requires_grad=False)
        init_ice_thick = start_surf - bed
        reference_model = Upstream2D(bed, dx=case.dx, mb_model=mb_model, y0=0,
                                     glen_a=cfg.PARAMS['glen_a'],
                                     ice_thick_filter=None,
                                     init_ice_thick=init_ice_thick)
        reference_model.run_until(y_end)
        return reference_model.surface_h.detach().numpy()


def compute_mean_bias(first_guess_surf, reference_surf, ice_mask):
    bias = (first_guess_surf - reference_surf).sum() / ice_mask.sum()
    # not totally exact, as first_guess_surf can (and probably will) not match ice_mask
    # exactly and the mean would have to be computed otherwise, but is ok for a first impression
    return bias


def iterative_first_guess(f, init_sca, reference_surf, ice_mask, dx,
                          mb_model, start_surf):
    max_iter = 100.
    sca = init_sca
    sca_step_size = 0.5
    sca_step = 0.
    min_sca = 1.5
    max_sca = 5.0

    first_bed_guess = first_guess(reference_surf, ice_mask, dx, factor=f,
                                  slope_cutoff_angle=sca)
    first_guess_surf = run_forward(mb_model, first_bed_guess, start_surf)
    bias = compute_mean_bias(first_guess_surf, reference_surf, ice_mask)
    print(bias)
    i = 1
    while abs(bias) > 5 and i < max_iter:
        if bias > 0:
            # need lower surface and therefore most probably also lower bed
            if sca_step > 0:
                # last step was positive as bias was negative -> reduce stepping
                sca_step_size *= 0.5
            sca_step = -sca_step_size
        else:
            if sca_step < 0:
                # last step was negative as bias was positive -> reduce stepping
                sca_step_size *= 0.5
            sca_step = sca_step_size
        sca = np.clip(sca + sca_step, min_sca, max_sca)

        i = i + 1
        first_bed_guess = first_guess(reference_surf, ice_mask, dx, factor=f,
                                      slope_cutoff_angle=sca)
        first_guess_surf = run_forward(mb_model, first_bed_guess, start_surf)
        bias = compute_mean_bias(first_guess_surf, reference_surf, ice_mask)
        print(bias)
        print(sca)
        if sca <= min_sca or sca >= max_sca:
            break  # TODO: do not break after first time this condition is met... bias could go into other direction again

    print('iteration finished')
    print(i)
    print(bias)
    print(sca)
    return first_bed_guess, sca, bias

first_bed_guess, sca, bias = iterative_first_guess(1., 5., reference_surf,
                                                   ice_mask, case.dx, mb,
                                                   start_surf)

ref_ice_mask = (reference_surf - bed_2d) == 0

ice_mask_for_plot = np.ma.masked_array(np.full(ice_mask.shape, 1),
                                       mask=ref_ice_mask)

masked_ice_thick_end = np.ma.masked_array(reference_surf - bed_2d,
                                          mask=ref_ice_mask)
masked_ice_thick_start = np.ma.masked_array(start_surf - bed_2d,
                                            mask=ref_ice_mask)
masked_reference_surf = np.ma.masked_array(reference_surf,
                                           mask=ref_ice_mask)

plt.ioff()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #see: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('terrain')
new_cmap = truncate_colormap(cmap, 0.3, 0.8)

f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('Bed of case {:s}, dx={:d}m'.format(case.name, case.dx))
fname = '{:s}_bed.png'.format(case.name)
plt.savefig(os.path.join(basedir, fname))
plt.clf()

f = plt.figure()
im_b = plt.imshow(first_bed_guess, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('First Bed Guess of case {:s}, dx={:d}m'.format(case.name, case.dx))
fname = '{:s}_first_guess.png'.format(case.name)
plt.savefig(os.path.join(basedir, fname))
plt.clf()


class MidpointNormalize(colors.Normalize):
    # see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


db = first_bed_guess - bed_2d
f = plt.figure()
im_b = plt.imshow(db, cmap='RdBu_r',
                  norm=MidpointNormalize(midpoint=0., vmin=db.min(),
                                         vmax=db.max()))
cbar = plt.colorbar(im_b)
cbar.set_label('$\Delta$Bed  height A.S.L (m)')
plt.title('Error of First Guess of case {:s}, dx={:d}m'.format(case.name,
                                                               case.dx))
fname = '{:s}_first_guess_error.png'.format(case.name)
plt.savefig(os.path.join(basedir, fname))
plt.clf()

f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('Surface of case {:s}, dx={:d}m, t={:d}a'.format(case.name, case.dx,
                                                          y_end))
plt.imshow(ice_mask_for_plot, 'binary', alpha=0.5)
fname = '{:s}_surf_y{:d}.png'.format(case.name, y_end)
plt.savefig(os.path.join(basedir, fname))
plt.clf()

cmap = plt.get_cmap('Blues_r')
cut_gray_cmap = truncate_colormap(cmap, 0.5, 0.99)

for y in [y_spinup_end, y_end]:
    f = plt.figure()
    im_b = plt.imshow(bed_2d, cmap=new_cmap)
    plt.title('Ice surface height, {:s}, dx={:d}m, t={:d}a'.format(
        case.name, case.dx, y))
    im_i = plt.imshow(masked_reference_surf, cut_gray_cmap)
    cbar = plt.colorbar(im_i)
    cbar.set_label('Surface height A.S.L. (m)')
    fname = '{:s}_surf_height_y{:d}.png'.format(case.name, y)
    plt.savefig(os.path.join(basedir, fname))
    plt.clf()

#For evaluation
gradients = np.gradient(bed_2d, case.dx)
gradients = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
angle = np.rad2deg(np.arctan(gradients))
print(np.max(angle))
print(np.min(angle))

plt.ion()

#inner_mask = torch.zeros(ice_mask.shape)
#inner_mask[1:-1, 1:-1] = torch.conv2d(torch.tensor([[ice_mask]], dtype=torch.float), torch.ones((1, 1, 3,3))) == 9
#plt.figure()
#plt.imshow(ice_mask - inner_mask)
#plt.show()