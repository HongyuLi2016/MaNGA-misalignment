#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_sample.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 01.08.2017
# Last Modified: 01.08.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import numpy as np
import util_sample_A as util_sample
import matplotlib.pyplot as plt
util_sample.boundary['zeta'][0] = 0.0
util_sample.boundary['eta'][0] = 0.0
mean_zeta = 0.75
mean_eta = 0.75
mean_Psai_int = 0.8
sigma_zeta = 0.1
sigma_eta = 0.1
sigma_Psai_int = 0.3
size = 300000
zeta = util_sample.get_sample(mean_zeta, sigma_zeta,
                              boundary=[0.5, 1.0], size=size)
eta = util_sample.get_sample(mean_eta, sigma_eta,
                             boundary=[0.5, 1.0], size=size)
Psai_int = util_sample.get_sample(mean_Psai_int, sigma_Psai_int,
                                  boundary=[0.0, np.pi/2.0], size=size)
fig, axes = plt.subplots(3, 1, figsize=(3, 6))
axes[0].hist(zeta, bins=100, range=[0.0, 1.0])
axes[1].hist(eta, bins=100, range=[0.0, 1.0])
axes[2].hist(Psai_int, bins=100, range=[0.0, np.pi/2.0])
fig.savefig('sample_A_scipy.png')
