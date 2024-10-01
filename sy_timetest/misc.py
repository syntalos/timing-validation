# -*- coding: utf-8 -*-
#
# Copyright (C) 2018-2020 Matthias Klumpp <matthias.klumpp@physiologie.uni-heidelberg.de>
#
# SPDX-License-Identifier: LGPL-3.0+

import os
import logging
import typing as T

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def plot_set_preferences(*, autolayout=True, sns_style='whitegrid', dark=False):
    """Set (my) preferences for plotting."""

    matplotlib.rcParams['figure.autolayout'] = autolayout
    plt.rc('axes.spines', top=False, right=False)
    sns.set()
    sns.set_style(sns_style)

    if dark:
        plt.style.use('dark_background')


def logging_set_basic_config(with_modname: bool = False):
    '''Set my preferred logging configuration.'''

    if with_modname:
        format_str = '%(asctime)s %(levelname)-6s %(name)s: %(message)s'
    else:
        format_str = '%(asctime)s %(levelname)-6s %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
