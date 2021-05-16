#! /usr/bin/env python

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from prcurve_singleiteration import get_confidence_from_pr, compute_f1score

# pickle file to interpolate
ffolder = '/home/dorian/Code/weed-detection/weed_detector/output/Tussock_v0_8/'
fname = 'Tussock_v0_8_prcurve.pkl'

fpath = ffolder + fname
with open(fpath, 'rb') as f:
    res = pickle.load(f)

print('res = ', res)

p = res['precision']
r = res['recall']
f1 = res['f1score']
ap = res['ap']
c = res['confidence']

# recompute f1score using p, r, not c:
f1score = compute_f1score(p, r)

# import code
# code.interact(local=dict(globals(), **locals()))

c_out = get_confidence_from_pr(p, r, c, f1score, pg=None, rg=0.95)

fig, ax = plt.subplots()
ax.plot(r, p, label='pr')
ax.plot(r, f1score, label='f1')
ax.plot(r, c, label='c')
ax.plot(r, c_out * np.ones((r.shape)), label='cout')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('pr-curve testing')
ax.legend()
# save_plot_name = os.path.join('output', _test_pr_final.png')
# plt.savefig(save_plot_name)
plt.show()


import code
code.interact(local=dict(globals(), **locals()))