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
p0 = [0.7, 0.5, 0.7, 0.5, 0.1, 0.1]
res = util_maxL.run_minimize(Psai_obs, eps_obs, p0)
sol = res['x']
print(sol)
print('Success: {}'.format(res['success']))
print(res['message'])
print('nit: {}'.format(res['nit']))
with open('{}/res.dat'.format(args[0]), 'w') as f:
    pickle.dump(res, f)
