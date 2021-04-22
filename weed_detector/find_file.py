#! /usr/bin/env python

import os

def find_file(file_pattern, folder):
    """
    find filename given file pattern in a folder
    """
    # TODO check valid inputs

    files = os.listdir(folder)
    file_find = []
    for f in files:
        if f.endswith(file_pattern):
            file_find.append(f)

    if len(file_find) <= 0:
        print('Warning: no files found matching pattern {}'.format(file_pattern))
    elif len(file_find) == 1:
        print('Found file: {}'.format(file_find[0]))
    elif len(file_find) > 1:
        print('Warning: found multiple files matching string pattern')
        for i, ff in enumerate(files_find):
            print('{}: {}'.format(i, ff))

    return file_find


# -----------__#
if __name__ == "__main__":

    patt = '.pth'
    folder = os.path.join('output', 'Tussock_v0_11')
    fname = find_file(patt, folder)
    print(fname)