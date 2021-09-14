#! /usr/bin/env python

"""
script to find area/centroid of polygon
"""

import os
import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np


# load annotation file
# extract the polygons
# plot the polygons
# compute centroid/area
# print results

dataset_name = 'Tussock_v4_poly286'

# folder locations and file names
root_dir = os.path.join('/home',
                        'dorian',
                        'Data',
                        'agkelpie',
                        dataset_name)

img_dir = os.path.join(root_dir, 'Images', 'Regions_Test')
ann_dir = os.path.join(root_dir, 'Annotations')

ann_file = 'via_project_6Aug2021_7h53m_json.json'
ann_path = os.path.join(ann_dir, ann_file)

# load annotations
ann_dict = json.load(open(ann_path))
ann_list = list(ann_dict.values())

# iterate through annotations, extract regions

img_names = []
region_areas = []
region_centroids = []
region_pts = []
for i, ann in enumerate(ann_list):
    img_name = ann['filename']
    img_names.append(img_name)
    regions = ann['regions']
    reg_areas = []
    reg_cent = []
    reg_pts = []
    if len(regions) > 0:
        for reg in regions:
            reg_type = reg['shape_attributes']['name']
            r = reg['shape_attributes']

            if reg_type == 'point':
                print('we have a point')
                cx = r['cx']
                cy = r['cy']
                area = 0
                reg_cent.append((cx, cy))
                reg_areas.append(area)
                reg_pts.append((cx, cy))

            elif reg_type == 'rect':
                print('we have a box')
                x_topleft = r['x']
                y_topleft = r['y']
                w = r['width']
                h = r['height']
                cx = x_topleft + w / 2.0
                cy = y_topleft + h / 2.0
                area = w * h
                reg_cent.append((cx, cy))
                reg_areas.append(area)

                box = [[x_topleft, y_topleft],
                       [x_topleft, y_topleft + h],
                       [x_topleft + w, y_topleft + h],
                       [x_topleft + w, y_topleft]]
                box = np.array(box)
                reg_pts.append(box)

            elif reg_type == 'polygon':
                print('we have a polygon')
                px = r['all_points_x']
                py = r['all_points_y']
                p = np.array([px, py]).T

                # https://math.stackexchange.com/questions/3177/why-doesnt-a-simple-mean-give-the-position-of-a-centroid-in-a-polygon
                poly = Polygon(p)
                cen = poly.centroid

                # import code
                # code.interact(local=dict(globals(), **locals()))
                cx = cen.coords[0][0]
                cy = cen.coords[0][1]
                # print(cen.coords)

                area = poly.area
                # print(area)
                reg_cent.append((cx, cy))
                reg_areas.append(area)
                reg_pts.append(p)
            else:
                print(f'warning: invalid region type: {reg_type}')

    region_areas.append(reg_areas)
    region_centroids.append(reg_cent)
    region_pts.append(reg_pts)


# verify?

import code
code.interact(local=dict(globals(), **locals()))

