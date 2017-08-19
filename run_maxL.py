#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: run_maxL.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 07.08.2017
# Last Modified: 07.08.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import util_maxL
import numpy as np
import util_angle
import util_sample_A as util_sample
import util_analysis
from optparse import OptionParser
import pickle
import os
# import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option('-f', action='store', type='string', dest='fname',
                  default='SR-catalogue.dat', help='obs data file name')
(options, args) = parser.parse_args()
if len(args) != 1:
    raise KeyError('please provide a folder name')
try:
    os.mkdir(args[0])
except OSError:
    pass

data = np.genfromtxt(options.fname, usecols=[1, 3])
eps_obs = data[:, 1]
Psai_obs = np.radians(data[:, 0])
# p0 = [0.8, 0.1, 0.85, 0.05, 0.78539816, 0.17453]
# p0 = [0.8, 0.1, 0.75, 0.15, 0.78539816, 0.1745]
p0 = [0.70710598, 0.08248679, 0.85836017, 0.13923725, 0.80944239, 0.14843667]
options = {'maxiter': 1000, 'disp': True, 'eps': 0.0001}
res = util_maxL.run_minimize(Psai_obs, eps_obs, p0, size=5000000, bins=25,
                             options=options, interp=True, seed=88562189)
sol = res['x']
print(sol)
print('Success: {}'.format(res['success']))
print(res['message'])
print('nit: {}'.format(res['nit']))
with open('{}/res.dat'.format(args[0]), 'w') as f:
    pickle.dump(res, f)

mean_zeta = sol[0]
mean_eta = sol[2]
mean_Psai_int = sol[4]
sigma_zeta = sol[1]
sigma_eta = sol[3]
sigma_Psai_int = sol[5]
# assign sampler parameters
size = 300000
# draw a sample for axis ratios and intrinsic misalignments
zeta = util_sample.get_sample(mean_zeta, sigma_zeta,
                              boundary=[0.5, 1.0], size=size)
eta = util_sample.get_sample(mean_eta, sigma_eta,
                             boundary=[0.5, 1.0], size=size)
ksai = eta * zeta
Psai_int = util_sample.get_sample(mean_Psai_int, sigma_Psai_int,
                                  boundary=[0.0, np.pi/2.0], size=size)
# calculate apparent misalignment
theta, phi = util_angle.get_view_angle(size)

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
