# -*- coding: utf-8 -*-
#
# Copyright (C) 2018-2023 Matthias Klumpp <matthias.klumpp@physiologie.uni-heidelberg.de>
#
# SPDX-License-Identifier: LGPL-3.0+

import os
import shutil
import typing as T
import logging as log
import tempfile
from pathlib import Path

import zarr
import numpy as np


def get_home_persistent_temp(subdir: os.PathLike) -> os.PathLike:
    """Obtain a persistent temporary location in the user's current home directory,
    or elsewhere if accessing HOME is not possible."""
    if os.path.isdir(Path.home()):
        location = os.path.join(Path.home(), '_scratch', subdir)
    else:
        location = os.path.join('/var/tmp', subdir)
    os.makedirs(location, exist_ok=True)
    return location


class CacheDirManager:
    """Create a directory to cache data in.

    Create a cache directory using a unique prefix (for the project) and cache key.
    The cache can either be automatically removed upon exit, or persisted, in which
    case data will be placed in the user's "~/_scratch" directory.

    This class can be used as a context manager.
    """

    def __init__(self, project_prefix: str, cache_key: str, *, delete_cache=True):
        self._unique_prefix = project_prefix
        self._cache_key = cache_key
        self._delete_cache = delete_cache
        self.temp_dir = None
        self._sys_tmp_dir = '/var/tmp'
        if not os.path.isdir(self._sys_tmp_dir):
            self._sys_tmp_dir = tempfile.gettempdir()

    def create(self) -> tuple[bool, os.PathLike]:
        """Create the cache and return whether it was empty and its location."""

        self.temp_dir = os.path.join(get_home_persistent_temp(self._unique_prefix), self._cache_key)
        if self._delete_cache:
            # cleanup after previous run, if necessary
            if os.path.exists(self.temp_dir):
                log.debug('Cleaning up old cache directory: {}'.format(self.temp_dir))
                shutil.rmtree(self.temp_dir)
            # use system temporary location if we delete the cache anyway
            self.temp_dir = tempfile.mkdtemp(prefix=self._unique_prefix + '_', dir=self._sys_tmp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        tmp_dir_empty = True
        with os.scandir(self.temp_dir) as it:
            if any(it):
                tmp_dir_empty = False

        if tmp_dir_empty:
            log.info(
                'Created new {} cache: {}:{}'.format(
                    'volatile' if self._delete_cache else 'persistent', self._unique_prefix, self._cache_key
                )
            )
        else:
            log.info('Reusing cached data for: {}:{}'.format(self._unique_prefix, self._cache_key))
        return tmp_dir_empty, Path(self.temp_dir)

    def cleanup(self):
        """Clean up the cache, remove it if we are allowed to do so."""
        if self.temp_dir and self._delete_cache:
            log.debug('Deleting unused cache directory: {}'.format(self.temp_dir))
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __del__(self):
        self.cleanup()

    def __enter__(self) -> tuple[bool, os.PathLike]:
        return self.create()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ZarrCache:
    """Manage a new cache that uses Zarr arrays as backing store."""

    def __init__(self, project_prefix: str, cache_name: str):
        """Create a new Zarr-backed cache. The project prefix is a constant, know prefix for a group
        of caches, while the cache_name is the unque name of an individual recording that we want
        to use to store the cache."""
        from numcodecs import Blosc

        self._cmgr = CacheDirManager(project_prefix, f'{cache_name}.zarr', delete_cache=False)

        cache_is_new, self._cache_fname = self._cmgr.create()
        self._cache_exists = not cache_is_new

        self._compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

    @property
    def exists(self) -> bool:
        """Returns True if the cache already existed."""
        return self._cache_exists

    def load(self, *, mode='r'):
        """Load data from cache, if it existed."""
        return zarr.open(self._cache_fname, mode=mode)

    def get_root(self):
        """Get the root of the Zarr storage."""

        # open Zarr-based cache with multiprocess synchronizer
        # IMPORTANT: Ensure Zarr archive is on filesystem which supports locks (no NFS!)
        synchronizer = zarr.ProcessSynchronizer(os.path.splitext(self._cache_fname)[0] + '.sync')
        zroot = zarr.open(self._cache_fname, mode='w', synchronizer=synchronizer)

        return zroot

    def replace(self, data: dict[str, T.Any]):
        """Replace all data in the cache with new data in :data."""
        zroot = self.get_root()

        for key, value in data.items():
            zroot[key] = zarr.array(value, compressor=self._compressor)
        self._cache_exists = True
