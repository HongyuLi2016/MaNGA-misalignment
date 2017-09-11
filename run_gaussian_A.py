#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : run_gaussian_A.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 08.09.2017
# Last Modified Date: 11.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
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
import util_sample_A as util_sample
import util_analysis
from optparse import OptionParser
import os
import pickle
import sys

util_sample.boundary['zeta'][0] = 0.6
util_sample.boundary['eta'][0] = 0.6

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
mean_zeta = 0.75
mean_eta = 0.75
mean_Psai_int = 1.2
sigma_zeta = 0.1
sigma_eta = 0.1
sigma_Psai_int = 0.1
# assign sampler parameters
nsteps = 1000
nwalkers = 300

means = np.array([mean_zeta, mean_eta, mean_Psai_int])
icov = np.zeros([3, 3])
icov[0, 0] = 1.0 / sigma_zeta**2
icov[1, 1] = 1.0 / sigma_eta**2
icov[2, 2] = 1.0 / sigma_Psai_int**2
paras = {}
paras['method'] = 'gaussian'
paras['hypers'] = {'means': means, 'icov': icov}
paras['steps'] = nsteps
paras['nwalkers'] = nwalkers
# draw a sample for axis ratios and intrinsic misalignments
hyperParas = util_sample.get_sample_mcmc(paras=paras)
# make figures about the sample
util_sample.analysis_sample(hyperParas, outpath=args[0])
# calculate apparent misalignment
flatchain = hyperParas['chain'].reshape(-1, 3)
theta, phi = util_angle.get_view_angle(nwalkers*nsteps)
zeta = flatchain[:, 0]
eta = flatchain[:, 1]
ksai = eta * zeta
Psai_int = flatchain[:, 2]
Psai, eps = util_angle.get_Psai(theta, phi, zeta, ksai, Psai_int)

# save resutls in a dict
rst = {}
rst['zeta'] = zeta
rst['eta'] = eta
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
