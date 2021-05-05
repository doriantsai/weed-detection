#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_confidence_threshold(prec, rec, conf, pgoal=None, rgoal=None):
    c = 0
    # TODO check valid input
    # check types
    # prec, rec, conf same lengths
    # if pgoal is not None and rgoal is None:
    #   valid P, find all P, choose best P
    # if rgoal is not None and pgoal is None:
    #   valid R, find all R, choose best R
    # if pgoal and rgoal not None
    #   choose closest and best PR

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
    # interp1d with fill_value set to extrapolate with a tuple might be best

    return c


rec = np.array([0.69072165, 0.72164948, 0.72164948, 0.72164948, 0.73195876,
       0.73195876, 0.74226804, 0.74226804, 0.75257732, 0.77319588,
       0.79381443])
prec_new = np.array([0.89333333, 0.83333333, 0.83333333, 0.83333333, 0.76344086,
       0.76344086, 0.70588235, 0.70588235, 0.64035088, 0.59055118,
       0.49677419])

rec_x = np.linspace(0, 1, num=101, endpoint=True)

# create prec_x by concatenating vectors first
prec_temp = []
rec_temp = []
for r in rec_x:
    if r < rec[0]:
        prec_temp.append(prec_new[0])
        rec_temp.append(r)

for i, r in enumerate(rec):
    prec_temp.append(prec_new[i])
    rec_temp.append(r)

for r in rec_x:
    if r >= rec[-1]:
        prec_temp.append(0)
        rec_temp.append(r)

prec_temp = np.array(prec_temp)
rec_temp = np.array(rec_temp)

# now interpolate:

# prec_x = np.interp(rec_x, rec_temp, prec_temp)
prec_interpolator = interp1d(rec_temp, prec_temp, kind='linear')
prec_x = prec_interpolator(rec_x)

# print(rec)
# print(prec_new)
# print(rec_x)
# print(prec_temp)
# print(rec_temp)

# import code
# code.interact(local=dict(globals(), **locals()))

fig, ax = plt.subplots()
ax.plot(rec_temp, prec_temp, color='blue', linestyle='dashed', label='combined')
ax.plot(rec, prec_new, marker='x', color='red', linestyle='solid', label='max-binned')
ax.plot(rec_x, prec_x, marker='o', color='green', linestyle='dotted', label='interp')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('prec-rec, interpolated')
ax.legend()
# save_plot_name = os.path.join('output', save_name, save_name + '_test_pr_extrap.png')
# plt.savefig(save_plot_name)
plt.show()

import code
code.interact(local=dict(globals(), **locals()))

# now for AP score, area under the graph of the interpolated pr-curve:
# ap = sum(Rn - Rn-1)*Pn
n = len(rec_x) - 1
ap = np.sum( (rec_x[1:n] - rec_x[0:n-1]) * prec_x[1:n] )

# now, we do interpolating
conf = np.linspace(0.1, 0.9, num=len(rec), endpoint=True)

# TODO
# specify what p, what r, given all p, all r, all conf
# first, make sure p, r are valid
# then output required conf to achieve p, r
# note: for extrapolation regions, take lowest/highest conf values
precision = prec_x
recall = rec_x
pgoal = 0.5
rgoal = 0.5
# choose the nearest?
c = compute_confidence_threshold(precision, recall, pgoal, rgoal, conf)

