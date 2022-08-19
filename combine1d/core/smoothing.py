import numpy as np
import copy
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR

import torch
from torch.autograd.function import Function


def signchange(ts):
    """Detect sign changes in a time series or numpy array.

    http://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-
    for-elements-in-a-numpy-array

    Returns
    -------
    An array with 0s everywhere and 1's when the sign changes
    """
    asign = np.sign(ts)
    sz = asign == 0
    # check if the given array contains only zeros
    if sz.all():
        return np.zeros(len(ts)).astype(int)
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0
    out = ((np.roll(asign, 1) - asign) != 0).astype(int)
    if asign[0] == asign[1]:
        out[0] = 0
    return out


def detect_instabilities(var, min_length=5, min_distance=3):
    """ This function searches for instabilities in a given variable. An
    instability is characterised as a zigzag pattern (signal is going up and
    down for a defined number of consecutive points with 'length_threshold').
    Moreover the minimum distance between two instabilities can be defined with
    'min_distance'.

    Parameters
    ----------
    var : ndarray
        Variable which should be checked for instabilities.
    min_length : int
        Defines the minimum number of grid points with a zigzag pattern before
        it is considered as an instability.
        Default: 5
    min_distance : int
        Defines the minimum number of grid points between two consecutive
        instabilities. All instabilities with a smaller distance between each
        other will be merged (could also merge more than two).
        Default: 3

    Returns
    -------
    instability_starts, instability_lengths : tuple of two lists
        First list contains the indices of instability starts, second list
        contains the corresponding lengths of the instabilities. If both are
        None no instability was detected.
    """
    # detect instabilities by a consecutive run of changing signs for the
    # difference of the input variable
    changing_sign = signchange(np.diff(var))

    # -------------------------
    # this part is adapted from
    # https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    n = changing_sign.shape[0]

    # find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(changing_sign[:-1], changing_sign[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run values
    run_values = changing_sign[loc_run_start]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, n))
    # -------------------------

    # sign changes are indicated with with run_values of 1, here also check the
    # minimum length (-1 because of staggered grid due to diff operation)
    index = np.logical_and(run_values == 1,
                           run_lengths >= min_length - 1)
    instability_starts = run_starts[index]

    # + 1 is needed as due to diff the changing_sign is on an staggered grid
    # and we want to include also the last grid point of the original variable
    instability_lengths = run_lengths[index] + 1

    if np.all(np.invert(index)):
        # Hurray no instabilities found
        return None, None
    elif np.sum(index) == 1:
        # Only one instability found so we can return it
        return instability_starts, instability_lengths
    else:
        # More than one instability was found, therefore we check if two
        # consecutive instabilities have at least the number of 'min_distance'
        # grid points in between, if not we combine them to one instability
        instability_ends = instability_starts + instability_lengths

        # the number of grid points between instabilities
        diff_instabilities = instability_starts[1:] - instability_ends[:-1]

        # here we use a loop as it could be that we must concatenate more
        # than two instabilities
        instability_starts_concat = []
        instability_lengths_concat = []
        start_tmp = None
        length_tmp = None
        for i, diff in enumerate(diff_instabilities < min_distance):
            if diff:
                if start_tmp is None:
                    start_tmp = instability_starts[i]
                    length_tmp = (instability_starts[i + 1] -
                                  instability_starts[i])
                else:
                    length_tmp += (instability_starts[i + 1] -
                                   instability_starts[i])
            else:
                if start_tmp is None:
                    start_tmp = instability_starts[i]
                    length_tmp = instability_lengths[i]
                else:
                    length_tmp += instability_lengths[i]
                instability_starts_concat.append(start_tmp)
                instability_lengths_concat.append(length_tmp)
                start_tmp = None
                length_tmp = None
        # and add last instability
        if diff:
            length_tmp += instability_lengths[-1]
            instability_starts_concat.append(start_tmp)
            instability_lengths_concat.append(length_tmp)
        else:
            instability_starts_concat.append(instability_starts[-1])
            instability_lengths_concat.append(instability_lengths[-1])

        return instability_starts_concat, instability_lengths_concat


def smoothe_instabilities(var, starts, lengths, smoothing_window=3,
                          smoothing_steps=15):
    '''

    Parameters
    ----------
    var
    starts
    lengths
    smoothing_window
    smoothing_steps

    Returns
    -------

    '''
    var_smoothed = copy.deepcopy(var)
    # smoothe each instability separately
    for num, (start, length) in enumerate(zip(starts, lengths)):

        end = start + length

        # be sure to do not exceed array
        start_add = start - smoothing_window
        start_smoothing_window = smoothing_window
        if start_add < 0:
            start_add = 0
            start_smoothing_window = start - start_add
        end_add = end + smoothing_window
        end_smoothing_window = smoothing_window
        if end_add > len(var):
            end_add = len(var)
            end_smoothing_window = end_add - end

        N = start_smoothing_window + end_smoothing_window + 1

        # points to smooth
        instability_smoothed = copy.deepcopy(var[start: end])

        # the actual smoothing happens here with a running mean
        for _ in range(smoothing_steps):
            instability_smoothed = np.append(var[start_add: start],
                                             instability_smoothed)
            instability_smoothed = np.append(instability_smoothed,
                                             var[end: end_add])
            instability_smoothed = np.convolve(instability_smoothed, np.ones(N) / N, mode='valid')

        var_smoothed[start: end] = instability_smoothed

    return var_smoothed


def smooth_variable(var):
    '''Efficient smoother'''
    window_width = 3

    #if window_width should be changeable need to adapt np.hstack of variable
    cumsum_vec = np.cumsum(
        np.hstack(
            (0,
             2 * var[0] - var[1],  # need to think about this limits if used again
             var,
             2 * var[-1] - var[-2])))
    var_smoothed = (cumsum_vec[window_width:] -
                    cumsum_vec[:-window_width]) / window_width

    return var_smoothed


def filter_spikes(var, std_limit=1.5, smoothing_window=3, smoothing_steps=2,
                  use_smoothing=False):
    """
    Filters out and smoothes spikes. A spike is defined as a value which is
    outside of mean +/- std_limit.

    Parameters
    ----------
    var
    std_limit
    smoothing_steps
    smoothing_window
    use_smoothing

    Returns
    -------

    """
    mean = np.mean(var)
    std = np.std(var)

    # first just clip all values outside of limits
    var_smoothed = np.clip(copy.deepcopy(var),
                           mean - std_limit * std,
                           mean + std_limit * std)

    if use_smoothing:
        # than smooth them to avoid to sharp edges
        bad_points = np.squeeze(np.argwhere(np.logical_or(
            var < mean - std_limit * std, var > mean + std_limit * std)),
            axis=1)

        if bad_points.size:
            for _ in range(smoothing_steps):
                for point in bad_points:
                    # be sure to do not exceed array
                    start_add = point - smoothing_window
                    start_smoothing_window = smoothing_window
                    if start_add < 0:
                        start_add = 0
                        start_smoothing_window = point - start_add
                    end_add = point + smoothing_window
                    end_smoothing_window = smoothing_window
                    if end_add > len(var):
                        end_add = len(var)
                        end_smoothing_window = end_add - point

                    N = start_smoothing_window + end_smoothing_window + 1

                    # points to smooth
                    point_smoothed = copy.deepcopy(var_smoothed[point])

                    # the actual smoothing happens here with a running mean
                    point_smoothed = np.append(var_smoothed[start_add: point],
                                               point_smoothed)
                    point_smoothed = np.append(point_smoothed,
                                               var_smoothed[point: end_add])
                    point_smoothed = np.convolve(point_smoothed, np.ones(N) / N,
                                                 mode='valid')

                    var_smoothed[point] = point_smoothed

    return var_smoothed


class smooth_velocity(Function):
    @staticmethod
    def forward(ctx, velocity, dt, max_dt, cfl_nr, dx, t):
        torch_type = velocity.dtype
        device = velocity.device

        velocity_np = velocity.detach().numpy().astype(np.float64)

        starts, lengths = detect_instabilities(velocity_np,
                                               min_length=5,
                                               min_distance=3)
        if starts:
            velocity_np = smoothe_instabilities(velocity_np,
                                                starts,
                                                lengths,
                                                smoothing_window=3,
                                                smoothing_steps=2)

        velocity_smoothed = filter_spikes(velocity_np,
                                          std_limit=1.5,
                                          use_smoothing=True)

        velocity_smoothed_torch = torch.tensor(velocity_smoothed,
                                               dtype=torch_type,
                                               device=device)

        # calculate smoothing factor for gradient calculation
        smoothing_factor = velocity / velocity_smoothed_torch
        smoothing_factor = torch.where(torch.isfinite(smoothing_factor),
                                       smoothing_factor,
                                       1.)
        smoothing_factor = torch.where(torch.abs(smoothing_factor) < 1.,
                                       1.,
                                       smoothing_factor)

        # if torch.any(smoothing_factor < 0):
        #    raise FloatingPointError('no negative smoothing factor allowed!')

        divisor = torch.max(torch.abs(velocity_smoothed_torch))
        if divisor == 0:
            dt_cfl = max_dt
        else:
            dt_cfl = cfl_nr * dx / divisor

        dt_use = torch.clamp(torch.min(dt_cfl, dt),
                             torch.tensor(0,
                                          dtype=torch_type,
                                          device=device,
                                          requires_grad=False),
                             max_dt)

        # save parameters for gradient calculation
        ctx.save_for_backward(smoothing_factor, (t + dt_use) / SEC_IN_YEAR,
                              velocity, velocity_smoothed_torch)

        return velocity_smoothed_torch

    @staticmethod
    def backward(ctx, grad_output):
        # use smoothing factor during AD
        smoothing_factor, t, velocity, velocity_smoothed = ctx.saved_tensors  # [0]

        # magnitude =
        if t % 1 == 0:
            print('just for fun')
        return grad_output / smoothing_factor, None, None, None, None, None


class smooth_slope(Function):
    @staticmethod
    def forward(ctx, slope, end_index, dt, max_dt, cfl_nr, dx, t, H_stag, rho, G, n,
                _fd, fs, sf_stag):
        torch_type = slope.dtype
        device = slope.device

        slope_np = slope.detach().numpy().astype(np.float64)

        starts, lengths = detect_instabilities(slope_np,
                                               min_length=5,
                                               min_distance=3)
        if starts:
            slope_smoothed = smoothe_instabilities(slope_np,
                                                   starts,
                                                   lengths,
                                                   smoothing_window=3,
                                                   smoothing_steps=2)
        else:
            slope_smoothed = slope_np

        slope_smoothed = filter_spikes(slope_smoothed,
                                       std_limit=1.5,
                                       use_smoothing=True)

        # exclude smoothing at the terminus
        slope_smoothed[end_index - 3: end_index] = slope_np[end_index - 3: end_index]

        slope_smoothed_torch = torch.tensor(slope_smoothed,
                                            dtype=torch_type,
                                            device=device)

        # calculate smoothing factor for gradient calculation
        smoothing_factor = slope / slope_smoothed_torch
        smoothing_factor = torch.where(torch.isfinite(smoothing_factor),
                                       smoothing_factor,
                                       1.)
        smoothing_factor = torch.where(torch.abs(smoothing_factor) < 1.,
                                       1.,
                                       torch.abs(smoothing_factor))
        smoothing_factor = smoothing_factor + 1.

        # if torch.any(smoothing_factor < 0):
        #    raise FloatingPointError('no negative smoothing factor allowed!')

        velocity = ((rho * G * slope) ** n *
                    (_fd * H_stag ** (n + 1) * sf_stag ** n +
                     fs * H_stag ** (n - 1)))
        velocity_smoothed = ((rho * G * slope_smoothed_torch) ** n *
                             (_fd * H_stag ** (n + 1) * sf_stag ** n +
                              fs * H_stag ** (n - 1)))
        divisor = torch.max(torch.abs(velocity_smoothed))
        if divisor == 0:
            dt_cfl = max_dt
        else:
            dt_cfl = cfl_nr * dx / divisor

        dt_use = torch.clamp(torch.min(dt_cfl, dt),
                             torch.tensor(0,
                                          dtype=torch_type,
                                          device=device,
                                          requires_grad=False),
                             max_dt)

        # save parameters for gradient calculation
        ctx.save_for_backward(smoothing_factor, (t + dt_use) / SEC_IN_YEAR,
                              slope, slope_smoothed_torch, velocity,
                              velocity_smoothed)

        return slope_smoothed_torch

    @staticmethod
    def backward(ctx, grad_output):
        # use smoothing factor during AD
        smoothing_factor, t, slope, slope_smoothed, velocity, velocity_smoothed = \
            ctx.saved_tensors  # [0]

        # magnitude =
        if t % 1 == 0:
            print('just for fun')
        return grad_output / smoothing_factor, None, \
               None, None, None, None, None, None, None, None, None, None, None, None
