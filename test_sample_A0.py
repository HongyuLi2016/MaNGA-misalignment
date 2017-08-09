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
import util_sample_A0 as util_sample
util_sample.boundary['zeta'][0] = 0.0
util_sample.boundary['ksai'][0] = 0.0
mean_zeta = 0.7
mean_ksai = 0.65
mean_Psai_int = 0.8
sigma_zeta = 0.1
sigma_ksai = 0.1
sigma_Psai_int = 0.3
means = np.array([mean_zeta, mean_ksai, mean_Psai_int])
icov = np.zeros([3, 3])
icov[0, 0] = 1.0 / sigma_zeta**2
icov[1, 1] = 1.0 / sigma_ksai**2
icov[2, 2] = 1.0 / sigma_Psai_int**2
paras = {}
paras['hypers'] = {'means': means, 'icov': icov}
paras['steps'] = 1000
rst = util_sample.get_sample_mcmc(paras=paras)
util_sample.analysis_sample(rst)
