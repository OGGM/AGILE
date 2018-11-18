import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_surrounding_line(exp_name, case, bed_measurements):
    lines = ()
    if exp_name == 'promised land 3c plus bed' and case == 'Giluwe':
        lines = (
        [[2.5, 15.5], [17.5, 15.5], [17.5, 7.5], [18.5, 7.5], [18.5, 15.5],
         [35.5, 15.5], [35.5, 16.5], [18.5, 16.5], [18.5, 27.5],
         [17.5, 27.5], [17.5, 16.5], [2.5, 16.5]],)
    if exp_name == 'promised land 3c plus bed' and case == 'Borden Peninsula':
        lines = ([[9.5, 13.5], [26.5, 13.5], [26.5, 14.5], [9.5, 14.5]],
                 [[9.5, 19.5], [26.5, 19.5], [26.5, 20.5], [9.5, 20.5]])
    if exp_name == 'identical-twin a plus bed' and case == 'Giluwe':
        lines = ([[15.5, 6.5], [18.5, 6.5]]
                 + [[18.5 + i + j, 7.5 + i] for i in range(3) for j in [0, 1]]
                 + [[21.5, 10.5], [18.5, 10.5]]
                 + [[18.5 - i - j, 9.5 - i] for i in range(3) for j in
                    [0, 1]],)
    return lines