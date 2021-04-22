#! /usr/bin/env python

"""
switch .pth file of model (at some hierarchy) with a snapshot of a specified epoch
"""

import os
import re
import numpy as np
import shutil

# TODO
# given a model name (.pth) and an epoch number
# find the .pth file of the model name
# find all the snapshots in the snapshots folder from training
# replace said .pth file with the nearest epoch?

def switch_snapshot(model_save_name, epoch):

    save_folder = os.path.join('output', model_save_name)
    save_path = os.path.join(save_folder, model_save_name + '.pth')
    print('Model save name: {}'.format(model_save_name))

    print('Searching for snapshot closest to epoch {}'.format(epoch))

    # find all filenames in snapshot folder
    snapshot_folder = os.path.join('output', model_save_name, 'snapshots')
    snapshot_files = os.listdir(snapshot_folder)
    pattern = 'epoch'
    e = []
    for f in snapshot_files:
        if f.endswith('.pth'):
            n = re.split(pattern, f, maxsplit=0, flags=0)
            e.append(int(n[1][:-4]))
    e = np.array(e)

    # find closest e[i] to epoch
    diff_e = np.sqrt((e - epoch)**2)
    i_emin = np.argmin(diff_e)

    # closest snapshot is indexed by i_emin
    print('closest snapshot epoch: {}'.format(snapshot_files[i_emin]))

    # remove original model
    print('removing {}'.format(save_path))
    os.remove(save_path)

    # copy/rename snapshot model
    snapshot_replacement = os.path.join(snapshot_folder, snapshot_files[i_emin])
    print('replacing {} with {}'.format(save_path, snapshot_replacement))
    replacement_path = os.path.join(save_folder, snapshot_files[i_emin])
    shutil.copyfile(snapshot_replacement, replacement_path)

    return snapshot_replacement


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # model to be replaced by a snapshot
    model_save_name =  'Tussock_v0_11'

    # snapshot:
    epoch = 25
    sr = switch_snapshot(model_save_name, epoch)

    # TODO:
    # consider not replacing the file, but simply making a pointer to the relevant snapshot?

    import code
    code.interact(local=dict(globals(), **locals()))

