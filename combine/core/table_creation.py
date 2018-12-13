import numpy as np
import salem
from combine.core.data_logging import load_pickle
from combine.core.arithmetics import mean_BIAS, RMSE
from combine.core.data_logging import DataLogger

def create_case_table(gdir):
    header = 'case,ela,mbgrad,zmax,sca,dx,V,A,hmean,hmax,coordinates\n'
    row = '{case:s},{ela_h:d},{mb_grad:g},{mb_max_alt:d},{sca:g}°,' \
          '{dx:d},{vol:.2f},{area:.2f},{mean_it:.1f},{max_it:.1f},' \
          '{{{coordinates}}}'

    vals = {}
    vals['case'] = gdir.case.name
    vals['ela_h'] = gdir.case.ela_h
    vals['mb_grad'] = gdir.case.mb_grad
    vals['mb_max_alt'] = gdir.case.mb_max_alt
    vals['dx'] = gdir.case.dx

    vals['sca'] = gdir.inversion_settings['fg_slope_cutoff_angle']
    it = np.load(gdir.get_filepath('ref_ice_thickness'))
    im = np.load(gdir.get_filepath('ref_ice_mask'))
    cell_area = gdir.case.dx**2
    vals['vol'] = cell_area * np.sum(it) * 1e-9
    vals['area'] = cell_area * np.sum(im) * 1e-6
    masked_it = np.ma.masked_array(it,
                                   mask=np.logical_not(im))
    vals['mean_it'] = np.mean(masked_it)
    #vals['min_it'] = np.min(masked_it)
    vals['max_it'] = np.max(masked_it)
    x = (gdir.case.extent[0, 0] + gdir.case.extent[1, 0])/2.
    y = (gdir.case.extent[0, 1] + gdir.case.extent[1, 1])/2.

    vals['coordinates'] = '{:g}$\\degree$W, {:g}$\\degree$N'.format(x, y)

    data_row = row.format(**vals)

    with open(gdir.get_filepath('casetable'), 'w') as f:
        f.writelines([header, data_row])

    return [header, data_row]


def eval_identical_twin(idir):
    header = 'case,run,icevolerr,rmsebed,rmsesurf,biasbed,biassurf,' \
             'corr,rmsefg,biasfg,iterations,' \
             'maxbeddiffglacier,maxbeddiffdomain,' \
             'minbeddiffglacier,minbeddiffdomain,' \
             'voloutsidebounds\n'
    row = '{case:s},{run:s},{dV:.2f},{rmsebed:.1f},{rmsesurf:.1f},' \
          '{biasbed:.1f},{biassurf:.1f},{corr:.3f},{rmsefg:.1f},' \
          '{biasfg:.1f},{iterations:d},' \
          '{maxbeddiffglacier:.1f},{maxbeddiffdomain:.1f},' \
          '{minbeddiffglacier:.1f},{minbeddiffdomain:.1f},' \
          '{voloutsidebounds:.9f}'
    # TODO: max_bed_diff?
    dl = load_pickle(idir.get_subdir_filepath('data_logger'))
    vals = {}
    vals['case'] = dl.case.name
    vals['run'] = idir.inv_settings['inversion_subdir']
    ref_it = np.load(idir.gdir.get_filepath('ref_ice_thickness'))
    mod_it = (dl.surfs[-1] - dl.beds[-1])
    ref_vol = ref_it.sum()
    mod_vol = mod_it.sum()
    vals['dV'] = (mod_vol - ref_vol) / ref_vol * 1e2
    ref_ice_mask = np.load(idir.gdir.get_filepath('ref_ice_mask'))
    vals['rmsebed'] = RMSE(dl.true_bed, dl.beds[-1], ref_ice_mask)
    vals['rmsesurf'] = RMSE(dl.ref_surf, dl.surfs[-1], ref_ice_mask)
    vals['rmsefg'] = RMSE(dl.true_bed, dl.first_guessed_bed, ref_ice_mask)

    vals['biasbed'] = mean_BIAS(dl.beds[-1], dl.true_bed, ref_ice_mask)
    vals['biassurf'] = mean_BIAS(dl.surfs[-1], dl.ref_surf, ref_ice_mask)
    vals['biasfg'] = mean_BIAS(dl.first_guessed_bed, dl.true_bed,
                               ref_ice_mask)

    masked_true_it = np.ma.masked_array(dl.ref_surf - dl.true_bed,
                                         mask=np.logical_not(ref_ice_mask))
    masked_mod_it = np.ma.masked_array(dl.surfs[-1] - dl.beds[-1],
                                         mask=np.logical_not(ref_ice_mask))
    # TODO: ice thickness
    vals['corr'] = np.ma.corrcoef(masked_true_it.flatten(),
                                  masked_mod_it.flatten())[0, 1]
    vals['iterations'] = len(dl.step_indices)
    #vals['maxbeddiff'] = np.max((dl.beds[-1] - dl.true_bed) * ref_ice_mask)
    vals['maxbeddiffglacier'] = np.max((dl.beds[-1] - dl.true_bed) *
                                         ref_ice_mask)
    vals['maxbeddiffdomain'] = np.max(dl.beds[-1] - dl.true_bed)
    #vals['minbeddiff'] = np.min((dl.beds[-1] - dl.true_bed) * ref_ice_mask)
    vals['minbeddiffglacier'] = np.min((dl.beds[-1] - dl.true_bed) *
                                        ref_ice_mask)
    vals['minbeddiffdomain'] = np.min(dl.beds[-1] - dl.true_bed)
    vals['voloutsidebounds'] = np.sum((dl.surfs[-1] - dl.beds[-1])
                                      * (1 - ref_ice_mask)) * 1e-9
    data_row = row.format(**vals)

    with open(idir.get_subdir_filepath('results'), 'w') as f:
        f.writelines([header, data_row])

    return [header, data_row]




