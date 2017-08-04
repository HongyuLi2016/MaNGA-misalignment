#!/usr/bin/env python
import os
import glob
py = glob.glob('*.py')
bin_dir = '/Users/lhy/astro_workspace/astro-packages/prolateMC'
for pyx in py:
    fname = pyx.split('/')[-1]
    print('diff for %s' % fname)
    os.system('diff %s %s/%s' % (pyx, bin_dir, fname))
