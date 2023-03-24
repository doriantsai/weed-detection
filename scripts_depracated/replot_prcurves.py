#! /usr/bin/env python

"""
script to replot pr-curves using the prcurve pkl files
"""

# specify prcurve files
#   we specify them via the model name, check existence first
# specify plot name
# read prcurve files
# output plot
# save plot
import os
import pickle
import glob
import matplotlib.pyplot as plt
from weed_detection.WeedModel import WeedModel as WM

# model names
# model_names = ['2021-03-25_MFS_Tussock_FasterRCNN_2021-08-31_22_21',
            #    '2021-03-25_MFS_Tussock_MaskRCNN_2021-08-31_19_33']
model_names = ['2021-03-26_MFS_Horehound_FasterRCNN_2021-09-09_20_17',
               '2021-03-26_MFS_Horehound_2021-09-09_18_08']
model_descriptions = ['Hh_FasterRCNN', 'Hh_MaskRCNN']
model_types = ['box', 'poly']
model_epochs = [25, 20]
models={'name': model_names,
        'folder': model_names,
        'description': model_descriptions,
        'type': model_types,
        'epoch': model_epochs}

datasets = ['blah', 'blah']

Horehound = WM()
Horehound.compare_models(models,
                         datasets,
                         load_prcurve=True,
                         show_fig=True)
# pickle file should be of the format:
# output/model_name/prcurve/model_name_prcurve.pkl
# legend_names = model_descriptions
# pr_dicts = []

# for m in model_names:
#     res = glob.glob(str('output/' + m+ '/prcurve/*_prcurve.pkl'))
#     # print(str('output/' + m + '/prcurve/*_prcurve.pkl'))
#     # print(res)
#     if len(res) <= 0:
#         print('ERROR: res not found')
#         print('ERROR: res = ' + res)
#     elif len(res) > 1:
#         print('ERROR: multiple res found, printing:')
#         for r in res:
#             print(r)
#     else:
#         # open pickle file
#         if os.path.isfile(res[0]):
#             with open(res[0], 'rb') as f:
#                 prd = pickle.load(f) # precision

#         # copmpute
#         pr_dicts.append(prd)


# # now, for recall = 80%, find the nearest confidence threshold:

# Tussock = WM()
# # do the plots
# fig, ax = plt.subplots()
# ap_list = []
# conf_goals = []
# p_goals = []
# for i, pr in enumerate(pr_dicts):
#     prec = pr['precision']
#     rec = pr['recall']
#     ap = pr['ap']
#     f1score = pr['f1score']
#     c = pr['confidence']
#     r_goal = 0.7
#     conf_goal = Tussock.get_confidence_from_pr(prec, rec, c, f1score, rg=r_goal)
#     p_goal = Tussock.get_p_from_r(prec, rec, r_goal)

#     conf_goals.append(conf_goal)
#     p_goals.append(p_goal)

#     # m_str = 'm={}, ap={:.2f}'.format(model_names[i], ap)
#     m_str = 'm={}, ap={:.2f}'.format(legend_names[i], ap)
#     ax.plot(rec, prec, label=m_str)
#     ap_list.append(ap)

# ax.legend()
# plt.xlabel('recall')
# plt.ylabel('precision')
# plt.title('model comparison: PR curve')
# plt.grid(True)

# mdl_names_str = "".join(legend_names)
# save_plot_name = os.path.join('output', 'model_compare_' +  mdl_names_str + '.png')
# plt.savefig((save_plot_name))

# plt.show()

# print('model comparison complete')
# for i, m in enumerate(model_names):
#     print(str(i) + ' model: ' + m)
#     print(str(i) + ' name: ' + legend_names[i])
#     print(f'{str(i)}: rgoal = {r_goal}')
#     print(f'{str(i)}: pgoal = {p_goals[i]}')
#     print(f'{str(i)}: conf = {conf_goals[i]}')


import code
code.interact(local=dict(globals(), **locals()))