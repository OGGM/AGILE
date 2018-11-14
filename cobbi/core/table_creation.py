import numpy as np

def create_case_table(gdir):
    header = 'case,ela,mbgrad,zmax,sca,dx,V,A,h,coordinates\n'
    row = '{case:s},${ela_h:d}$,${mb_grad:g}$,${mb_max_alt:d}$,${sca:g}°$,' \
          '${dx:d}$,${vol:.2f}$,${area:.2f}$,\\shortstack{{${mean_it:.1f}$ ' \
          '\\\\(${max_it:.1f}$ / ${min_it:.1f}$)}},{{{coordinates}}}'

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
    vals['min_it'] = np.min(masked_it)
    vals['max_it'] = np.max(masked_it)
    vals['coordinates'] = '\\shortstack{{${:g}$°W, ${:g}$°N to  \\ \\' \
                          '${:g}$°W, ${:g}$°N}}'.format(
        gdir.case.extent[0, 0],
        gdir.case.extent[0, 1],
        gdir.case.extent[1, 0],
        gdir.case.extent[1, 1])

    data_row = row.format(**vals)

    with open(gdir.get_filepath('casetable'), 'w') as f:
        f.writelines([header, data_row])

    return [header, data_row]