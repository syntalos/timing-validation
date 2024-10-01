#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:26:31 2021

@author: Martin Both & Matthias Klumpp
"""

import os
import math
import logging as log

import pint
import edlio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from scipy import stats
from scipy.fft import fft, ifft
from edlio.dataio.tsyncfile import TSyncFileMode
from matplotlib.collections import PatchCollection

from syntalos_timetest.ttutils import (
    SyntalosTimeSyncDataLoader,
    align_start_times,
)
from syntalos_timetest.misc import (
    plot_set_preferences,
    logging_set_basic_config,
)
from syntalos_timetest.cachemgr import get_home_persistent_temp
from syntalos_timetest.syntalos_tsexp_defs import (
    EXPERIMENTS_MARATHON,
    EXPERIMENTS_LAUNCHSYNC,
)

# %%

# root location where your data is
DATA_ROOT_DIR = '/path/to/my/recorded/data'

# select the current experiment type, either 'marathon' or 'launchsync'
CURRENT_EXPERIMENT_CLASS = 'marathon' # 'launchsync'
CURRENT_EXPERIMENT_IDX = -1  # use the last experiment fro the list

# to-be-loaded entry index in the respective experiment list
# whether caches should be loaded, if they exist
USE_CACHED = True
# whether Intan data should be synchronized, or if synchronization should
# be skipped entirely (this should always be True)
SYNC_INTAN_TS = True


# %% collection of definitions

# time base of Syntalos is in microseconds, time base of the analysis should be in ms
t_base = 1000

# these are the offset times we have measured for our computer and our devices with the launch sync analysis part with respect to the events
#             ["intan-raw", "intan", "events", "gvid", "tisvid", "mscope", "arvvid", "tisarvvid", "micro-events"]
offset_list = [0.35, -9.40, -0.05, -30.55, -39.84, -25.04, -37.22, -20.46, -50.71]  # reference events (=2)

# this is for illustration purpuses: the length of one frame in ms if applicable
l_array = [
    "intan raw",
    "intan sync",
    "events",
    "gvid",
    "tisvid",
    "mscope",
    "arvvid",
    "tisarvvid",
    "micro events",
]  # labels
rec_duration = [1, 1, 1, 40, 15, 33, 40, 15, 1]  # recording duration for each frame in ms

#  historical reasons why this is hard coded
intan_sampling_rate = 20000


# ################################

HAVE_LATENCY_DATA = True
CURRENT_EXPERIMENT_CLASS = CURRENT_EXPERIMENT_CLASS.lower()
if CURRENT_EXPERIMENT_CLASS == 'marathon':
    CURRENT_EXPERIMENT = EXPERIMENTS_MARATHON[CURRENT_EXPERIMENT_IDX]
    HAVE_LATENCY_DATA = CURRENT_EXPERIMENT.get('pylatency', True)
    EDL_DIR = '{}/{}/{}'.format(
        DATA_ROOT_DIR, CURRENT_EXPERIMENT['date'], CURRENT_EXPERIMENT.get('ename', 'SyncTest-1')
    )
elif CURRENT_EXPERIMENT_CLASS == 'launchsync':
    CURRENT_EXPERIMENT = EXPERIMENTS_LAUNCHSYNC[CURRENT_EXPERIMENT_IDX]
    EDL_ROOT = '{}/{}'.format(
        DATA_ROOT_DIR, CURRENT_EXPERIMENT['date']
    )
else:
    raise RuntimeError('Experiment class "{}" is unknown.'.format(CURRENT_EXPERIMENT_CLASS))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHES_DIR = get_home_persistent_temp('syntalos-timing-playground')
GRAPHICS_OUT_DIR = os.path.join(SCRIPT_DIR, 'SyntalosSyncTestGraphics', CURRENT_EXPERIMENT['date'])

# get default Pint unit registry
ureg = pint.get_application_registry()

# configure logging
logging_set_basic_config()

# configure plotting defaults
plot_set_preferences()


def get_onsetTimes(datIn, thr):
    iHigh = np.nonzero(datIn[1] > 0.5)[0]
    dHigh = np.diff(np.concatenate(([0], iHigh)))
    dHigh = np.nonzero(dHigh > thr)[0]
    iOn = np.array(iHigh)[dHigh.astype(int)]
    tOn = datIn[0, iOn]
    return tOn


def get_onsetTimes_intanraw(datInTime, datInValues, thr):
    iHigh = np.nonzero(datInValues > 0.5)[0]
    dHigh = np.diff(np.concatenate(([0], iHigh)))
    dHigh = np.nonzero(dHigh > thr)[0]
    iOn = np.array(iHigh)[dHigh.astype(int)]
    tOn = datInTime[iOn]
    return tOn


def load_intan_tsync_aux_data(dset):
    start_offset_usec = 0
    sync_map = np.empty([0, 2])

    for tsf in dset.read_aux_data():
        if tsf.sync_mode != TSyncFileMode.SYNCPOINTS:
            raise Exception(
                'Can not synchronize RHD signal timestamps using a tsync file that is not in \'syncpoints\' mode.'
            )
        if tsf.time_units != (ureg.usec, ureg.usec):
            raise Exception(
                'For RHD signal synchronization, both timestamp units in tsync file must be microseconds. Found: {}'.format(
                    tsf.time_units
                )
            )
        sync_map = np.vstack((sync_map, tsf.times))
    start_offset_usec = sync_map[0][0] - sync_map[0][1]
    return sync_map, start_offset_usec


def interpolate_start_time(tOn_expected, tOn_data):
    if len(tOn_expected) < len(tOn_data):
        tOn_data = tOn_data[0 : len(tOn_expected)]
    if len(tOn_expected) > len(tOn_data):
        tOn_expected = tOn_expected[0 : len(tOn_data)]

    _, intercept, _, _, _ = stats.linregress(tOn_expected, tOn_data)
    return intercept


def med_filt(dat_in, med_len=9):
    filt_mat = np.zeros((len(dat_in) + med_len, med_len))
    for cL in range(0, med_len):
        filt_mat[cL : len(dat_in) + cL, cL] = dat_in
    dat_filt = np.median(filt_mat, axis=1)
    return dat_filt[int(np.floor(med_len / 2)) : int(np.floor(med_len / 2) + len(dat_in))]


def get_onset_tables(intan_raw, timings, t_base):
    tOn_IntanR = get_onsetTimes(intan_raw[0:2], thr=10) / t_base
    tOn_IntanC = get_onsetTimes(timings['intan'][0:2], thr=10) / t_base
    tOn_tisvid = get_onsetTimes(timings['tisvid'][0:2], thr=1) / t_base
    tOn_gvid = get_onsetTimes(timings['gvid'][0:2], thr=1) / t_base
    tOn_mscope = get_onsetTimes(timings['mscope'][0:2], thr=1) / t_base
    tOn_arvvid = get_onsetTimes(timings['arvvid'][0:2], thr=1) / t_base
    tOn_Events = get_onsetTimes(timings['events'][0:2], thr=1) / t_base
    tOn_tisarvvid = get_onsetTimes(timings['arvvid-2'][0:2], thr=1) / t_base
    tOn_micro_Events = get_onsetTimes(timings['micro-events'][0:2], thr=1) / t_base

    log.info('Onset times evaluated')

    # FIND THE FIRST COMMON EVENT

    t_array = np.array(
        [
            tOn_IntanR[0:5],
            tOn_IntanC[0:5],
            tOn_Events[0:5],
            tOn_gvid[0:5],
            tOn_tisvid[0:5],
            tOn_mscope[0:5],
            tOn_arvvid[0:5],
            tOn_tisarvvid[0:5],
            tOn_micro_Events[0:5],
        ]
    )

    i_start = np.zeros(
        (len(t_array)),
    )

    device_last = np.argmax(t_array[:, 0])
    for cD in range(0, t_array.shape[0]):
        i_start[cD,] = np.argmin(abs(t_array[cD, :] - t_array[device_last, 0]))

    log.info('first common event identified')

    # now put all the onset times in a big array called t_array
    t_array = [
        tOn_IntanR[i_start[0].astype(int) :],
        tOn_IntanC[i_start[1].astype(int) :],
        tOn_Events[i_start[2].astype(int) :],
        tOn_gvid[i_start[3].astype(int) :],
        tOn_tisvid[i_start[4].astype(int) :],
        tOn_mscope[i_start[5].astype(int) :],
        tOn_arvvid[i_start[6].astype(int) :],
        tOn_tisarvvid[i_start[7].astype(int) :],
        tOn_micro_Events[i_start[8].astype(int) :],
    ]

    # and set them to the same length for all devices
    t_len = np.nan * np.ones((len(t_array),))
    for cM in range(0, len(t_array)):
        if len(t_array[cM]) > 0:
            t_len[cM] = len(t_array[cM])

    t_len = np.nanmin(t_len).astype(int)
    for cM in range(0, len(t_array)):
        if len(t_array[cM]) > 0:
            t_array[cM] = t_array[cM][0:t_len]

    log.info('event timetable generated')

    # Intan raw, Intan sync, events, gvid, tisvid, mscope, arvvid, tisarvvid, microevents
    return t_array, device_last, i_start


def get_stim_array(t_array):
    stim_diffs = (np.round(np.diff(t_array[0]))).astype(int)
    min_time = t_array[0][0]  # Intan is the fastest clock
    stim_array = np.cumsum(np.concatenate((min_time * np.ones((1,)), stim_diffs)))

    return stim_array


def get_deviations(t_array):
    stim_array = get_stim_array(t_array)
    d_array = []
    for cD in range(0, len(t_array)):
        d_array.append(t_array[cD] - stim_array)

    return d_array


def fft_filter(X, sampling_rate, flo, fhi, order=8):

    if len(X.shape) != 1:
        if X.shape[1] > 1:
            raise Exception('fft filter only works with 1-d vectors')
        else:
            X = X.reshape((X.size,))

    flag_unevenlength = 0
    if (np.floor(len(X) / 2) * 2) != len(X):
        print('fft filter only works with even number of entries. discarding the last value')
        X = X[0 : (np.floor(len(X) / 2) * 2).astype(int)]
        flag_unevenlength = 1

    nsample = len(X)
    period = nsample / sampling_rate
    hzpbin = 1 / period

    i = np.arange(1, nsample / 2 + 2)  # %left one-sided spectrum->from nyquist frequency to zero

    if flo == 0:
        factor_lo0 = np.ones(int(nsample / 2 + 1))
    else:
        r_lo = np.power(((i - 1) * hzpbin / flo), (2 * order))  # %->normalized spectrum: hzpbin/flo=n
        factor_lo0 = r_lo / (1 + r_lo)

    if fhi > sampling_rate:
        factor_hi0 = np.ones(int(nsample / 2 + 1))
    else:
        r_hi = np.power(((i - 1) * hzpbin / fhi), (2 * order))
        factor_hi0 = 1 / (1 + r_hi)

    factor_lo = np.concatenate((factor_lo0, factor_lo0[-2:0:-1]))  # fliplr(factor_lo0(2:end-1))
    factor_hi = np.concatenate((factor_hi0, factor_hi0[-2:0:-1]))

    fftx = fft(X)
    if len(fftx) > len(factor_lo):
        factor_lo = np.concatenate((factor_lo0, factor_lo0[-1:0:-1]))
    if len(fftx) > len(factor_hi):
        factor_hi = np.concatenate((factor_hi0, factor_hi0[-1:0:-1]))
    fftx_l = fftx * np.sqrt(factor_lo * factor_hi)
    L = ifft(fftx_l)

    L_return = L.real
    if flag_unevenlength == 1:
        L_return = np.concatenate((L_return, (L_return[-1],)))

    return L_return


def skyline_plot(data_x, data_y, axes_handle):
    import numpy as np

    xV = np.reshape(np.reshape(np.concatenate((data_x, data_x)), (2, len(data_x))).T, (1, 2 * len(data_x)))
    yV = np.reshape(
        np.reshape(
            np.concatenate(
                (
                    np.concatenate((np.reshape(data_y[0], (1,)), data_y)),
                    np.concatenate((data_y, np.reshape(data_y[0], (1,)))),
                )
            ),
            (2, len(data_x)),
        ).T,
        (1, 2 * len(data_x)),
    )

    axes_handle.plot(xV[0], yV[0])


# %%

if __name__ == '__main__':
    # %%
    # NOTE: Example of how to process marathon experiment data - will only work
    # if we have one such experiment selected
    # %%

    if CURRENT_EXPERIMENT_CLASS.startswith('launchsync'):

        # historical reasons: it was more robust to start the analysis after a few seconds. so we start here at the 8th event
        iStart_offset = 8

        # historical reasons: to be able to omit one of the experiments or start at a certain experiment
        max_run = 100
        start_run = 40
        take_out = np.array([])

        num_experiments = np.min((len(CURRENT_EXPERIMENT['runs']), max_run)) - start_run - len(take_out)
        num_devices = len(l_array)

        onset_times_array = np.zeros(
            (num_devices, num_experiments)
        )  # the real time stamp difference of the first fram with respect to the intan time stamp
        start_delays_array = np.zeros(
            (num_devices, num_experiments)
        )  # the interpolated estimated start of the devices
        round_time_array = np.zeros((0, 3))  # the round times
        cE = -1
        cA = -1

        for run_name in CURRENT_EXPERIMENT['runs']:
            cA += 1

            if cA >= max_run:
                log.info('maximal number of experiments reached')
                break

            if cA < start_run:
                log.info('start run not yet reached')
                continue

            if any(cA == take_out):
                log.info('broken run taken out')
                continue

            cE += 1

            # get the data
            edl_dir = os.path.join(EDL_ROOT, run_name)
            tsdl = SyntalosTimeSyncDataLoader(edl_dir, cache_dir=CACHES_DIR)
            tsdl.use_cached = USE_CACHED

            # the colors we use for the different devices
            color_list = [
                tsdl.color_for('intan-raw'),  # intanR
                tsdl.color_for('intan'),  # intanC
                tsdl.color_for('events'),  # events
                tsdl.color_for('gvid'),  # gvid
                tsdl.color_for('tisvid'),  # tisvid
                tsdl.color_for('mscope'),  # mscope
                tsdl.color_for('arvvid-1'),  # arvvid
                tsdl.color_for('arvvid-2'),  # tisarvvid
                tsdl.color_for('micro-events'),  # micro-Events
            ]

            log.info(
                'Loading data for LaunchSync experiment {}, Intan sync: {}, use caches: {}'.format(
                    tsdl.dcoll.collection_idname,
                    'enabled' if SYNC_INTAN_TS else 'disabled',
                    'yes' if USE_CACHED else 'no',
                )
            )
            log.info(
                'Recording length was: {} msec ({:.2f} min)'.format(
                    tsdl.recording_length.to(ureg.msec), tsdl.recording_length.to(ureg.min)
                )
            )

            timings, intan_raw = tsdl.load(SYNC_INTAN_TS)

            # ensure all entriest start at the right starting time
            timings = align_start_times(timings)

            # get the onset times (i.e. the time points when the TTL pulse/the LED light is detected) of all devices
            t_array, device_last, i_start = get_onset_tables(intan_raw, timings, t_base)

            # and the deviations to the expected offsets
            d_array = get_deviations(t_array)

            # estimated start delays: interpolate enough onset times to get a good estimate of the 'real' timestamp delay the device gets assigned by Syntalos
            # this delay is constant for a certain combination of computer and devices
            # historical reasons: it seemed more robust to take points in the middle of the recording
            start_points = 50
            num_points = 150
            x_vals = np.arange(0, num_points)
            for cD in range(0, len(d_array)):
                start_delays_array[cD, cE] = interpolate_start_time(
                    x_vals, d_array[cD][start_points : start_points + num_points]
                )

            # this table is for plotting the time window of the frame in which the first event of a recording was detected by the different devices
            stim_array = get_stim_array(t_array)
            for cD in range(0, len(d_array)):
                onset_times_array[cD, cE] = t_array[cD][0] - stim_array[0]

            # the round trip times for the Arduino and the Pi Pico
            # intan_raw[0,:] = the time stamps
            # intan_raw[1,:] = the original TTL pulse
            # intan_raw[2,:] = the Arduino TTL pulse
            # intan_raw[3,:] = the Pi Pico TTL pulse
            tOn_stim = get_onsetTimes_intanraw(intan_raw[0, :], intan_raw[1, :], thr=1) / t_base
            tOn_round_ard = get_onsetTimes_intanraw(intan_raw[0, :], intan_raw[2, :], thr=1) / t_base
            tOn_round_mpy = get_onsetTimes_intanraw(intan_raw[0, :], intan_raw[3, :], thr=1) / t_base

            # again: start at the first common event
            tOn_stim = tOn_stim[tOn_stim >= t_array[0][0]]
            tOn_round_ard = tOn_round_ard[tOn_round_ard >= t_array[0][0]]
            tOn_round_mpy = tOn_round_mpy[tOn_round_mpy >= t_array[0][0]]

            # and put it into one table for easier plotting
            tOn_array = np.concatenate(
                (
                    np.reshape(tOn_stim, (len(tOn_stim), 1)),
                    np.reshape(tOn_round_ard, (len(tOn_stim), 1)),
                    np.reshape(tOn_round_mpy, (len(tOn_stim), 1)),
                ),
                axis=1,
            )
            round_time_array = np.concatenate((round_time_array, tOn_array))

        # %% now that we collected all the data, we can do the plots

        # %% Fig 5A: example of start time estimation
        c_ind = -1
        p_ind = np.array([2, 3, 4, 5, 7, 8])  # for this figure, plot only some devices
        number_columns = 2
        fig, ax = plt.subplots(nrows=3, ncols=number_columns)
        for cD in range(2, num_devices):
            if all(np.invert(p_ind == cD)):
                continue
            c_ind += 1
            i1 = math.floor(c_ind / number_columns)
            i2 = c_ind % number_columns

            plot_values = d_array[cD][start_points : start_points + num_points]
            estimated_start = np.median(plot_values)
            ax[i1, i2].plot(
                stim_array[start_points : start_points + num_points] / 1000,
                plot_values,
                '.',
                color=color_list[cD],
            )
            ax[i1, i2].plot(
                [stim_array[start_points] / 1000, stim_array[start_points + num_points] / 1000],
                np.array([1, 1]) * estimated_start,
                color='#FF0000',
            )
            ax[i1, i2].plot(stim_array[start_points] / 1000, plot_values[0], 'o', color='#FF0000')
            ax[i1, i2].set_ylim((-15, 45))

        log.info('Figure 5A: example of start time estimation')

        # %% Figure 5B:
        num_bins = 20
        wid_bins = 0.25
        markersize = 10
        number_columns = 2
        fig1, ax1 = plt.subplots(nrows=3, ncols=number_columns)
        fig2, ax2 = plt.subplots(nrows=1, ncols=len(p_ind))
        fig1_size = [24.72, 7.31]
        fig2_size = [14.4, 2.04]
        fig1.set_size_inches(fig1_size)
        fig2.set_size_inches(fig2_size)
        c_ind = -1
        for cP in range(2, num_devices):
            if all(np.invert(p_ind == cP)):
                continue
            c_ind += 1
            i1 = math.floor(c_ind / number_columns)
            i2 = c_ind % number_columns

            plot_values = start_delays_array[cP, :]
            med_value = np.round(np.mean(plot_values))
            left_edge = med_value - np.floor(num_bins / 2 * wid_bins)
            right_edge = med_value + np.floor(num_bins / 2 * wid_bins)

            ax1[i1, i2].plot(plot_values, '.', color=color_list[cP], markersize=markersize)
            ax1[i1, i2].set_title('Figure 5C: ' + l_array[cP])
            ax1[i1, i2].set_ylim([left_edge, right_edge])

            hVals, bEdges, _ = ax2[c_ind].hist(
                plot_values,
                bins=num_bins,
                range=(left_edge, right_edge),
                facecolor=color_list[cP],
                edgecolor=color_list[cP],
            )
            ax2[c_ind].set_title('Figure 5C: ' + l_array[cP])
            ax2[c_ind].set_ylim([0, 30])

        log.info('Figure 5B: estimated onset times: scatter plots and estimated onset times: histograms')

        # %% Figure 5 C once with raw times, once with known delay subtracted
        frame_len = [
            1,
            1,
            1,
            1000 / 25,
            1000 / 60,
            1000 / 30,
            1000 / 25,
            1000 / 60,
            1,
        ]  # frames length of the individual devices in ms
        delay_list = np.zeros((2, num_devices))  # to move the the onset time by the known delays
        for cD in range(2, num_devices):
            delay_list[1, cD] = np.mean(start_delays_array[cD, :]) + frame_len[cD] / 2
        scl_fac = 10  # arbitrary scaling factor
        scl_bar = 50  # scale bar in ms
        num_trials = 7  # display how many trials?
        start_trial = 0  # start with which trial?
        for cT in range(0, 2):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(
                np.array([num_trials - 1, num_trials - 1 + scl_bar / 1000 * scl_fac]),
                np.array([1, 1]) * (-0.5),
                color='#000000',
            )
            for cP in range(0, num_trials):
                ax.plot(
                    np.array([1, 1]) * cP, np.array([0, num_devices]) - 0.25, color=np.array([1, 1, 1]) * 0.8
                )
                for cD in range(0, num_devices):
                    frame_time = onset_times_array[cD, cP + start_trial] - delay_list[cT, cD]
                    if frame_len[cD] == 1:
                        ax.plot(
                            np.array([1, 1]) * cP + frame_time / 1000 * scl_fac,
                            np.array([0, 0.5]) + cD,
                            color=color_list[cD],
                        )
                    else:
                        plot_boxes = []
                        plot_boxes.append(
                            pch.Rectangle(
                                (cP + frame_time / 1000 * scl_fac, cD), frame_len[cD] / 1000 * scl_fac, 0.25
                            )
                        )
                        pc = PatchCollection(plot_boxes, edgecolor='none', facecolor=color_list[cD])
                        ax.add_collection(pc)

        log.info('Figure 5C: Ticks of the first frame for each trial')

        # %% Figure 5D:
        num_bins = [100, 10]
        left_edge = [0, 0]
        right_edge = [10, 0.25]

        r_array = ['Arduino', 'MicroPy']

        for cP in range(0, 2):
            fig, ax = plt.subplots()
            plot_values = round_time_array[:, cP + 1] - round_time_array[:, 0]
            hVals, bEdges, _ = ax.hist(
                plot_values,
                bins=num_bins[cP],
                range=(left_edge[cP], right_edge[cP]),
                facecolor=color_list[cP],
                edgecolor=color_list[cP],
            )
            ax.set_title(
                'Figure 5D: roundtime of '
                + r_array[cP]
                + '. Max time: '
                + str(np.round(np.max(plot_values), 3))
                + ' ms'
            )

        log.info('Figure 5D: Arduino and MicroPy round time')

    # %%
    if CURRENT_EXPERIMENT_CLASS.startswith('marathon'):
        #    do_marathon_experiment_stuff()

        # %%
        tsdl = SyntalosTimeSyncDataLoader(EDL_DIR, cache_dir=CACHES_DIR)
        tsdl.use_cached = USE_CACHED

        log.info(
            'Analyzing data for experiment from {}, Intan sync: {}, use caches: {}'.format(
                tsdl.experiment_date_str,
                'enabled' if SYNC_INTAN_TS else 'disabled',
                'yes' if USE_CACHED else 'no',
            )
        )

        # ensure graphics output dir exists
        log.info('Graphics export dir is: {}'.format(GRAPHICS_OUT_DIR))
        os.makedirs(GRAPHICS_OUT_DIR, exist_ok=True)

        timings, intan_raw = tsdl.load(SYNC_INTAN_TS)

        intan_newts_m = timings['intan'][0]

        # # %%
        sig_m = np.concatenate(
            (
                intan_newts_m.reshape((len(intan_newts_m), 1)).T,
                timings['intan'][1].reshape((len(timings['intan'][1]), 1)).T,
            ),
            axis=0,
        )

        # get the onset times (i.e. the time points when the TTL pulse/the LED light is detected) of all devices
        t_array, device_last, i_start = get_onset_tables(intan_raw, timings, t_base)

        # and the deviations to the expected offsets
        d_array = get_deviations(t_array)

        # %% now that we have collected the data, lets plot the figures
        # figure 3C: Just the time ticks
        window_beg = [100, len(t_array[0]) - 100]
        window_len = 5
        reference_channel = 1
        yOff = 25
        xVals = np.arange(0, len(t_array[0])) * 1000

        fig0, ax0 = plt.subplots(nrows=1, ncols=len(window_beg))
        ax0[0].set_title('Figure 3C')

        for cW in range(0, len(window_beg)):
            for cP in range(0, len(d_array)):
                xVals = t_array[cP][int(window_beg[cW]) : int(window_beg[cW]) + window_len] + offset_list[cP]
                yVals = np.ones(len(xVals)) * cP
                if rec_duration[cP] > 1:
                    plot_boxes = []
                    for cL in range(0, window_len):
                        plot_boxes.append(pch.Rectangle((xVals[cL], yVals[cL]), rec_duration[cP], 0.5))
                    pc = PatchCollection(plot_boxes, edgecolor='none', facecolor=color_list[cP])
                    ax0[cW].add_collection(pc)
                else:
                    for cL in range(0, window_len):
                        ax0[cW].plot(
                            [xVals[cL], xVals[cL]], [yVals[cL], yVals[cL] + 0.5], color=color_list[cP]
                        )
                if cP == reference_channel:
                    for cL in range(0, window_len):
                        xRef = t_array[cP][int(window_beg[cW] + cL)] + offset_list[cP]
                        ax0[cW].plot([xRef, xRef], [-0.2, 8.7], linestyle='--', color=np.ones(3) * 0.6)

        log.info('Figure 3C: time ticks')

        # %% figure 3D: plot the onset times minus the expected onset times
        # expected means that we define the signal generator clock as the 'true clock'

        x_limits = [[-100, len(t_array[0]) + 100], [100, 450], [len(t_array[0]) - 500, len(t_array[0]) - 150]]
        y_limits = [[-25, 850], [-25, 175], [650, 850]]
        for cF in range(0, 3):
            xVals = np.arange(0, len(t_array[0]))
            fig1, ax1 = plt.subplots()
            for cP in range(0, len(d_array)):
                if len(t_array[cP]) > 0:
                    ax1.plot(
                        xVals,
                        np.round(d_array[cP], decimals=2),
                        label=l_array[cP],
                        color=color_list[cP],
                    )

            ax1.legend()
            if cF == 0:
                ax1.set_title('Figure 3D')
                log.info('Figure 2D: event onsets recorded vs. expected over the whole time range')
            if cF == 1:
                ax1.set_title('Figure 3E at the beginning of recording')

            if cF == 2:
                ax1.set_title('Figure 3E at the end of recording')

            ax1.set_xlim(x_limits[cF])
            ax1.set_ylim(y_limits[cF])

        log.info('Figure 3E: close up of 3D')

        # %% figure 4A:
        log.info('Figure 4A: shifting means of event onsets')
        # analysis of the variability  of the internal pc-clock

        # first, quantify the linear time shift of the computer clock with respect to the expected times from the signal generator
        # (it is just basic maths for a linear equation with two points given)
        # reference device for the linear time shift is the event channel
        ref_shift = 2
        # and we define two points (plus window around them) from which the linear time shift is quantified
        ref_ind_beg = 8000
        ref_ind_end = 80000
        ref_win = 100
        # this gives us the following points:
        px_1 = int(ref_ind_beg + 0.5 * ref_win)
        py_1 = np.mean(d_array[ref_shift][ref_ind_beg : ref_ind_beg + ref_win])
        px_2 = int(ref_ind_end + 0.5 * ref_win)
        py_2 = np.mean(d_array[ref_shift][ref_ind_end : ref_ind_end + ref_win])
        # and the linear regression:
        m = (py_2 - py_1) / (px_2 - px_1)
        c = py_1 - (m * px_1)
        linear_timeshift = m * np.arange(0, len(d_array[0])) + c

        # smooth (filter) the reference channel. historical reasons, why each device can be filtered with individual filter settings
        # ["intan-raw", "intan", "events", "gvid", "tisvid", "mscope", "arvvid", "tisarvvid", "micro-events"]
        fHiList = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

        min_val = np.inf
        max_val = -np.inf

        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        for cP in range(0, len(d_array)):
            device_time = d_array[cP] - linear_timeshift
            device_time = device_time[0 : (np.floor(len(device_time) / 2) * 2).astype(int)]
            device_filt = fft_filter(
                device_time,
                sampling_rate=2000,
                flo=0,
                fhi=fHiList[cP],
                order=2,
            )

            device_offset = 0
            if cP > 0:
                device_offset = np.mean(device_filt[ref_ind_end : ref_ind_end + ref_win])

            ax5.plot(device_filt - device_offset, label=l_array[cP], color=color_list[cP])
            if cP > 0:
                ax4.plot(device_filt - device_offset, label=l_array[cP], color=color_list[cP])

        ax5.legend()
        ax5.plot([0, len(d_array[6])], [0, 0], '0.5')
        ax5.set_xlim(800, len(d_array[0]) - 750)
        ax5.set_title('Figure 4A')

        ax4.legend()
        ax4.plot([0, len(d_array[6])], [0, 0], '0.5')
        ax4.set_xlim(800, len(d_array[0]) - 750)
        ax4.set_title('Figure 4A')

        # %% figure 4B and figure 4C: in 6 seperate figures:
        # jitter of event onset times compared to a defined reference device
        # measured at thre time points: at the beginning, in the middle, and at the end
        # parameters of the analysis windows, in seconds
        # used as INDICES(!) in tArray, as the events are spaced by
        # one second, each

        window_beg = [1000, len(d_array[0]) - 2000]
        if window_beg[0] > window_beg[1]:
            window_beg = [len(d_array[0]) - 2500, len(d_array[0]) - 2500]
        window_wid = 800
        if window_wid > int(np.round(np.max(t_array[0]) / 1000)):
            window_wid = int(np.round(np.max(t_array[0]) / 1000 * 0.6))

        log.info('Figure 4B')
        log.info('Figure 4C in 6 separate figures')

        ax_list = ['none'] * len(d_array)
        fig_list = ['none'] * len(d_array)
        xVals = np.arange(0, len(d_array[0]))

        dat_ref = d_array[ref_shift] - d_array[device_last][0] + yOff
        dat_ref_filt = fft_filter(dat_ref, sampling_rate=2000, flo=0, fhi=fHiList[ref_shift], order=2)

        # inds = [2, 3, 5, 4, 0, 1, 6, 7, 8]

        y_range = 300
        y_lower = [1000, 1000, 1000]

        y_scale = np.zeros([len(window_beg), 9])
        y_value = np.zeros([len(window_beg), 9])
        x_left = np.zeros([len(window_beg), 9])
        x_right = np.zeros([len(window_beg), 9])

        t2 = ' ms'
        t4 = 'width: '
        t5 = 'total time shift: '

        nBins = 720
        bBeg = -120
        bEnd = 240

        fig_traces, ax_traces = plt.subplots(nrows=1, ncols=len(window_beg))

        for cA in range(len(ax_list)):
            fig_list[cA], ax_list[cA] = plt.subplots()

        # and now plot the figures

        for cP in range(0, len(d_array)):

            dat_comp = d_array[cP] - d_array[device_last][0] + yOff + offset_list[cP]

            for cW in range(0, len(window_beg)):
                w_beg = window_beg[cW]
                w_end = window_beg[cW] + window_wid
                # w_plo = int(window_wid / 4)
                dat_part = dat_comp[w_beg:w_end] - dat_ref_filt[w_beg:w_end]

                ax_traces[cW].plot(np.arange(w_beg, w_end), dat_part, color=color_list[cP])
                ax_traces[cW].set_ylim([-40, 40])

                mVal = np.mean(dat_part)

                bVals = np.linspace(bBeg, bEnd, nBins + 1) + (bEnd - bBeg) / nBins / 2
                hVals, bEdges = np.histogram(dat_part, bins=nBins, range=(bBeg, bEnd))

                skyline_plot(bEdges, hVals, ax_list[cP])

                ax_list[cP].plot([mVal, mVal], [0, max(hVals)], color='r')
                ax_list[cP].set_title(l_array[cP])

                y_scale[cW, cP] = max(hVals)
                y_value[cW, cP] = mVal
                x_left[cW, cP] = np.min(dat_part)
                x_right[cW, cP] = np.max(dat_part)

            t6 = str(round(np.abs(y_value[0, cP] - y_value[-1, cP]), 2))
            ax_list[cP].text(-20, (0.9) * max(y_scale[:, cP]), t5 + t6 + t2)

            for cW in range(0, len(window_beg)):
                t1 = str(round(y_value[cW, cP], 2))
                ax_list[cP].text(-20, (0.5 + 0.1 * cW) * max(y_scale[:, cP]), t1 + t2)
                t3 = str(round(x_right[cW, cP] - x_left[cW, cP]))
                ax_list[cP].text(-20, (0.1 + 0.1 * cW) * max(y_scale[:, cP]), t4 + t3 + t2)

            ax_list[cP].set_xlim([-25, 25])

            if cP == 0:
                ax_list[cP].set_xlim([-25, 175])

        # %% Figure 4D: jitter of sampling rate due to syntalos synchronization
        log.info('Figure 4D: jitter of sampling rate due to syntalos synchronization')
        t_beg = 40000
        t_end = 42000
        t_beg = len(d_array[0]) - 2500
        t_end = len(d_array[0]) - 500

        # time differences between the sampling points
        dat_d = np.diff(intan_newts_m[t_beg * intan_sampling_rate : t_end * intan_sampling_rate])  # µs
        # mean time differnde
        dat_dm = np.mean(dat_d)  # µs

        # find the time points, where sampling intervals change
        # change in interval
        diff_diff = np.diff(dat_d)
        # indices and values at the time points, the interval changes
        step_inds = np.where(np.invert((diff_diff < 4e-5) & (diff_diff > -4e-5)))[0]
        step_vals = dat_d[step_inds]

        fig8, ax8 = plt.subplots()
        ax8.cla()
        skyline_plot(np.concatenate((np.zeros((1,)), step_inds)), step_vals, ax8)

        ax8.plot([0, len(dat_d)], [dat_dm, dat_dm])
        t1 = str(np.round(np.mean(dat_d), 9))
        ax8.text(0, np.max(dat_d), t1)

        # Fig 4E: length distribution and maximum shift distribution

        sample_diff = np.diff(step_inds)
        time_diff = sample_diff / intan_sampling_rate

        fig10, ax10 = plt.subplots()

        nBins = 200
        bBeg = 49.99
        bEnd = 50.01

        hVals, bEdges = np.histogram(dat_d, bins=nBins, range=(bBeg, bEnd))
        plot_boxes = []
        for cL in range(0, len(hVals)):
            plot_boxes.append(pch.Rectangle((bEdges[cL], 0), (bEdges[cL + 1] - bEdges[cL]), hVals[cL]))
        pc = PatchCollection(plot_boxes, edgecolor=color_list[0], facecolor=color_list[0])
        ax10.add_collection(pc)
        ax10.plot([dat_dm, dat_dm], [0, np.max(hVals)], '#ff0000')
        ax10.set_xlim(bBeg, bEnd)
        ax10.set_ylim(0, 1.1 * np.max(hVals))
        t1 = str(dat_dm)
        ax10.text(50.0005, 1.05 * np.max(hVals), t1)

        fig13, ax13 = plt.subplots()

        nBins = 119
        bBeg = 1
        bEnd = 120

        hVals, bEdges = np.histogram(time_diff, bins=nBins, range=(bBeg, bEnd))
        plot_boxes = []
        for cL in range(0, len(hVals)):
            plot_boxes.append(pch.Rectangle((bEdges[cL], 0), (bEdges[cL + 1] - bEdges[cL]), hVals[cL]))
        pc = PatchCollection(plot_boxes, edgecolor=color_list[0], facecolor=color_list[0])
        ax13.add_collection(pc)
        ax13.set_xlim(bBeg, bEnd)
        ax13.set_ylim(0, 1.1 * np.max(hVals))
