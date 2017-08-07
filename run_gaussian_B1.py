#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: run_gaussian.py
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
import util_angle
import util_sample_B as util_sample
import util_analysis
from optparse import OptionParser
import os
import pickle
import sys

# util_angle.boundary['zeta'][0] = 0.0
# util_angle.boundary['eta'][0] = 0.0

parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) != 1:
    print 'Error - please provide a folder name'
    exit(1)
try:
    os.mkdir(args[0])
except OSError:
    pass
f = open('{}/stdout'.format(args[0]), 'w')
sys.stdout = f
# assign hyper parameter values
mean_zeta = 0.7
mean_eta = 0.7
mean_Psai_int = 0.7853982
sigma_zeta = 0.5
sigma_eta = 0.5
sigma_Psai_int = 1.0
# assign sampler parameters
nsteps = 1000
nwalkers = 300
size = nsteps*nwalkers
# draw a sample for axis ratios and intrinsic misalignments
zeta = util_sample.get_sample(mean_zeta, sigma_zeta,
                              boundary=[0.5, 1.0], size=size)
eta = util_sample.get_sample(mean_eta, sigma_eta,
                             boundary=[0.5, 1.0], size=size)
ksai = eta * zeta
Psai_int = util_sample.get_sample(mean_Psai_int, sigma_Psai_int,
                                  boundary=[0.0, np.pi/2.0], size=size)
# calculate apparent misalignment
theta, phi = util_angle.get_view_angle(nwalkers*nsteps)

Psai, eps = util_angle.get_Psai(theta, phi, zeta, ksai, Psai_int)

# save resutls in a dict
rst = {}
rst['zeta'] = zeta
rst['ksai'] = ksai
rst['Psai_int'] = Psai_int
rst['Psai'] = Psai
rst['eps'] = eps
rst['costheta'] = np.cos(theta)
rst['phi'] = phi
# analysis distribution of apparent misalignment
util_analysis.analysis_distribution(rst, outpath=args[0])
# util_analysis.plot_axis_ratio(zeta, ksai, outpath=args[0])
with open('{}/rst.dat'.format(args[0]), 'w') as f:
    pickle.dump(rst, f)
f.close()
