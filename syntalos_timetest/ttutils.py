#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:08:11 2020

Common utility functions for reading and preprocessing timing data
produced by Syntalos timing evaluation and latency test experiments.
This variant also parses data from the new "devclock" module added
to the test setup in December 2020.

@author: Matthias Klumpp
"""

import os
import sys
import logging as log
import traceback
import multiprocessing as mp

import cv2 as cv
import pint
import zarr
import edlio
import numpy as np
from numba import jit

# get default Pint unit registry
ureg = pint.get_application_registry()


@jit(nopython=True, nogil=True)
def first_ge_index(vec, c):
    '''Get the index of the first greater-equal number to `c`
    in ascending-sorted vector `vec`'''
    for i in range(len(vec)):
        if vec[i] >= c:
            return int(i)
    return int(0)


def read_intan_timings(dset, have_latency_data, do_timesync=True):
    log.info('Reading Intan signal data...')

    x_time = []
    x_time_noadj = []
    y_sig = []
    y_sig_response1 = []
    y_sig_response2 = []

    for intan in dset.read_data(include_nosync_time=True, do_timesync=do_timesync):
        x_time.append(intan.sync_times)
        x_time_noadj.append(intan._nosync_ts)
        y_sig.append(intan.digin_channels_raw[0] * 1)
        if have_latency_data:
            y_sig_response1.append(intan.digin_channels_raw[1] * 1)
            if len(intan.digin_channels_raw) > 2:
                y_sig_response2.append(intan.digin_channels_raw[2] * 1)

    x_time = np.concatenate(x_time)
    x_time_noadj = np.concatenate(x_time_noadj)
    y_sig = np.concatenate(y_sig)
    y_sig_response1 = np.concatenate(y_sig_response1)
    y_sig_response2 = np.concatenate(y_sig_response2)

    log.info('Intan signal data loading & adjustment completed.')
    if not (x_time_noadj.size == x_time.size == y_sig.size == y_sig_response1.size):
        raise Exception(
            (
                'Dimensions of read intan data make no sense: '
                f'{x_time_noadj.size=}, {x_time.size=}, {y_sig.size=}, {y_sig_response1.size=}'
            )
        )
    return x_time_noadj, x_time, y_sig, y_sig_response1, y_sig_response2


def read_video_signal(dset, led_style='old', detect_radius=4):
    log.info('Processing raw video data for "{}"...'.format(dset.name))

    res_x = []
    res_y = []

    if led_style == 'old':
        brightness_threshold = 80000
    else:
        brightness_threshold = 360000

    # check if LED is on (or not)
    last_frame_time = -1 * 1000 * 1000 * ureg.usec
    for i, frame in enumerate(dset.read_data()):
        if frame.time < last_frame_time:
            log.error(
                'Frame timestamp progression is nonsensical: {} <= {}'.format(frame.time, last_frame_time)
            )
            assert frame.time >= last_frame_time
        last_frame_time = frame.time
        res_x.append(frame.time.to(ureg.usec).magnitude)

        mat_green = frame.mat.copy()
        mat_green[:, :, 0] = 0
        mat_green[:, :, 2] = 0

        mask = cv.inRange(mat_green, (0, 40, 0), (0, 255, 0))
        mask_halfmax = np.amax(mask) // 4
        frame_bright_value = mask[mask > mask_halfmax].sum()
        led = 1 if frame_bright_value > brightness_threshold else 0

        res_y.append(led)

    log.info('Analysis for "{}" video data completed.'.format(dset.name))
    return np.asarray(res_x) * ureg.usec, np.asarray(res_y)


def read_miniscope_signal(dset):
    log.info('Processing miniscope data for "{}"...'.format(dset.name))

    res_x = []
    res_y = []

    # check if LED is on (or not)
    last_frame_time = -1 * ureg.usec
    for i, frame in enumerate(dset.read_data()):
        if frame.time < last_frame_time:
            log.error(
                'Frame timestamp progression is nonsensical: {} <= {}'.format(frame.time, last_frame_time)
            )
            assert frame.time >= last_frame_time
        last_frame_time = frame.time
        res_x.append(frame.time.to(ureg.usec).magnitude)

        led_visible = 1 if np.mean(frame.mat) > 100 else 0
        res_y.append(led_visible)

    log.info('Analysis for "{}" miniscope data completed.'.format(dset.name))
    return np.asarray(res_x) * ureg.usec, np.asarray(res_y)


def read_events_table(dset):
    log.info('Reading events table...')

    x = []
    y = []
    for row in dset.read_data():
        if row[0] == 'RecTime' or row[0] == 'Time':
            continue
        x.append(int(row[0]))
        y.append(int(row[1]))

    log.info('Events table read.')
    return np.asarray(x) * ureg.usec, np.asarray(y)


def read_microevents_data(dset):
    log.info('Reading micro-events data...')

    df = list(dset.read_data())[0]

    if 'timestamp_msec' in df:
        return np.asarray(df['timestamp_msec']) * ureg.msec, np.asarray(df['Data'])
    else:
        return np.asarray(df['timestamp_usec']) * ureg.usec, np.asarray(df['Data'])


def trim_by_times(x, y, time_start, time_end):
    idx_start = -1
    idx_end = -1
    for i, v in enumerate(x):
        if idx_start < 0 and v >= time_start:
            idx_start = i
        if idx_end < 0 and v >= time_end:
            idx_end = i
            break

    if idx_start < 0 or idx_end < 0:
        return x, y
    return x[idx_start:idx_end], y[idx_start:idx_end]


def magnitude(v):
    '''Return magnitude for any value'''
    if hasattr(v, 'magnitude'):
        return v.magnitude
    return v


def trim_by_time(x, y, time_start):
    idx_start = first_ge_index(x, time_start)
    return np.asarray((x[idx_start:], y[idx_start:]))


def align_start_times(timings):
    '''Determine starting time, when the slowest of the devices has
    acquired its first datapoint'''
    start_time = max([magnitude(e[0][0]) for e in timings.values()])
    for k, v in timings.items():
        timings[k] = trim_by_time(magnitude(v[0]), v[1], start_time)
    return timings


def plot_tsync_testpoint(ax, dset, title, color=None, fix_intan=False):
    timedata = np.empty([0, 2])

    first = True
    for ts in dset.read_data():
        times = ts.times
        if first:
            first = False
            if fix_intan:
                times = times[1:, :]
        timedata = np.vstack((timedata, times))

    ax.set_title(title)
    ax.set_xlabel('Master Time [µs]')
    ax.set_ylabel('Offset [µs]')
    ax.plot(timedata[:, 0][:-2], timedata[:, 1][:-2], color=color)


def mp_read_data(zroot, edl_dir, idname, have_latency_data, led_style, intan_timesync_enabled):
    '''Dispatch to raw data loading methods and store generated arrays in Zarr store'''

    try:
        dcoll = edlio.load(edl_dir)
        videos_group = dcoll.group_by_name('videos')

        if idname == 'intan':
            dset = dcoll.dataset_by_name('intan-signals')
            x_time_noadj, x_time, y_sig, y_sig_response1, y_sig_response2 = read_intan_timings(
                dset, have_latency_data, do_timesync=intan_timesync_enabled
            )

            # unadjusted timestamps, recorded input signal, Python feedback signal
            zroot['intan-raw'] = np.asarray(
                (x_time_noadj.to(ureg.usec).magnitude, y_sig, y_sig_response1, y_sig_response2)
            )

            # adjusted timestamps with recorded input signal
            zroot[idname] = np.asarray((x_time.to(ureg.usec).magnitude, y_sig))
            return

        data = None
        if idname == 'events':
            dset = dcoll.dataset_by_name('events')
            if dset:
                data = read_events_table(dset)
            else:
                log.warning('No Arduino event data found!')
        elif idname == 'micro-events':
            dset = dcoll.dataset_by_name('micro-events')
            if dset:
                data = read_microevents_data(dset)
            else:
                log.warning('No MicroPython Workbench data found!')
        elif idname == 'gvid':
            dset = videos_group.dataset_by_name('generic-camera')
            if not dset:
                log.warning('No UVC Webcam data found!')
                return None
            data = read_video_signal(dset, led_style)
        elif idname == 'tisvid':
            dset = videos_group.dataset_by_name('tis-camera')
            if dset:
                data = read_video_signal(dset, led_style, detect_radius=2)
            else:
                log.warning('No ImagingSource Camera data found!')
        elif idname == 'arvvid':
            dset = videos_group.dataset_by_name('aravis-camera')
            if dset:
                data = read_video_signal(dset, led_style)
        elif idname == 'arvvid-1':
            dset = videos_group.dataset_by_name('aravis-camera-1')
            if dset:
                data = read_video_signal(dset, led_style)
            else:
                log.warning('Data from Aravis Camera 1 not found!')
        elif idname == 'arvvid-2':
            dset = videos_group.dataset_by_name('aravis-camera-2')
            if dset:
                data = read_video_signal(dset, led_style)
            else:
                log.warning('Data from Aravis Camera 2 not found!')
        elif idname == 'mscope':
            dset = videos_group.dataset_by_name('miniscope')
            if not dset:
                log.warning('No Miniscope data found!')
                return None
            data = read_miniscope_signal(dset)
        else:
            raise ValueError('Unknown ID name: {}'.format(idname))

        if data is not None:
            zroot[idname] = np.asarray((data[0].to(ureg.usec).magnitude, data[1]))

    except Exception as e:
        tb = traceback.format_exc()
        return e, tb
    return None


class SyntalosTimeSyncDataLoader:
    '''
    Loads and preprocesses all data associated with the time-sync experiment
    and also handles automatic, transparent caching.
    The data itself will not be modified, this class oly returns raw data.
    '''

    def __init__(self, edl_dir, *, cache_dir=None):
        if not cache_dir:
            from makutils import get_home_persistent_temp

            cache_dir = get_home_persistent_temp('sy-tsdl')

        self._cache_root = cache_dir
        self._use_cached = True
        self.have_latency_data = True
        self.intan_timesync_enabled = True

        self._edl_dir = edl_dir
        self.dcoll = edlio.load(edl_dir)
        self.videos_group = self.dcoll.group_by_name('videos')
        self.intern_group = self.dcoll.group_by_name('syntalos_internal')

        self.label_map = {
            'intan': ['Intan', '#27ae60'],
            'intan-raw': ['Intan (unadjusted)', '#808000'],
            'events': ['Events (Py)', '#dd8452'],
            'micro-events': ['Pico Pi Events', '#707d8a'],
            'gvid': ['UVC Webcam', '#483d8b'],
            'tisvid': ['TIS Camera', '#4c72b0'],
            'arvvid': ['Aravis Camera', '#1e90ff'],
            'arvvid-1': ['Aravis Camera 1', '#1e90ff'],
            'arvvid-2': ['Aravis Camera 2', '#00ffff'],
            'mscope': ['Miniscope', '#9b59b6'],
        }

    @property
    def use_cached(self) -> bool:
        '''Returns whether we should use caches or reload data'''
        return self._use_cached

    @use_cached.setter
    def use_cached(self, v: bool):
        '''Set whether we should use caches or reload data'''
        self._use_cached = v

    def load(self, sync_intan_timestamps=True, *, led_style='old'):
        # we set global state here, which is ugly but the only way we can ensure
        # our preferred defaults are actually set (other start methods will not work
        # well in IPython notebooks)
        if mp.get_start_method() != 'forkserver':
            mp.set_start_method('forkserver', force=True)

        os.makedirs(self._cache_root, exist_ok=True)
        cache_fname = os.path.join(
            self._cache_root, 'syt-{}_rawsig.zarr'.format(self.dcoll.collection_idname)
        )
        if self._use_cached and not os.path.exists(cache_fname):
            log.warning('Can not load cached data as requested, because no cache exists. Regenerating.')
            self._use_cached = False

        if not self._use_cached:
            if os.path.exists(cache_fname):
                import shutil

                shutil.rmtree(cache_fname)

            pool = mp.Pool(mp.cpu_count() if mp.cpu_count() < 8 else 8)
            log.info('Preprocessing raw data, recreating cache.')

            # open Zarr-based cache with multiprocess synchronizer
            # IMPORTANT: Ensure Zarr archive is on filesystem which supports locks (no NFS!)
            synchronizer = zarr.ProcessSynchronizer(os.path.splitext(cache_fname)[0] + '.sync')
            zroot = zarr.open(cache_fname, mode='w', synchronizer=synchronizer)

            data_idnames = [
                'intan',
                'events',
                'micro-events',
                'gvid',
                'tisvid',
                'arvvid',
                'arvvid-1',
                'arvvid-2',
                'mscope',
            ]
            results = [
                pool.apply_async(
                    mp_read_data,
                    (
                        zroot,
                        self._edl_dir,
                        idname,
                        self.have_latency_data,
                        led_style,
                        self.intan_timesync_enabled,
                    ),
                )
                for idname in data_idnames
            ]
            # check for any errors and rethrow them if necessary
            for res in results:
                res = res.get()
                if res is not None:
                    print(res[1], file=sys.stderr)
                    raise res[0]

            pool.close()
            pool.join()

        log.info('Opening Zarr array store (read-only)')
        timings = dict(zarr.open(cache_fname, mode='r'))
        intan_raw = timings.pop('intan-raw')

        # replaced sync data with unsynchronized data if Intan
        # sync is switched off
        if not sync_intan_timestamps:
            timings['intan'] = np.asarray((intan_raw[0], intan_raw[1]))

        return timings, intan_raw

    def internal_dataset(self, name: str):
        return self.intern_group.dataset_by_name(name)

    def dataset(self, name: str):
        return self.dcoll.dataset_by_name(name)

    def color_for(self, idname: str):
        return self.label_map[idname][1]

    def label_for(self, idname: str):
        return self.label_map[idname][0]

    @property
    def recording_length(self):
        return (self.dcoll.attributes['recording_length_msec'] * 1000.0) * ureg.usec

    @property
    def experiment_date_str(self):
        return self.dcoll.time_created.strftime('%y-%m-%d')
