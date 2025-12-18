import argparse
from ast import arg
import cv2
import copy
import numpy as np
import os
import sys

from yaml import parse
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR))
from utils import read_points, read_calib, read_label, bbox_camera2lidar, vis_pc, bbox3d2corners,\
     points_lidar2image, vis_img_3d


def vis_gt(root, id, saved_root): 
    img_path = os.path.join(root, 'image_2', f'{id}.png')
    lidar_path = os.path.join(root, 'velodyne', f'{id}.bin')
    calib_path = os.path.join(root, 'calib', f'{id}.txt') 
    label_path = os.path.join(root, 'label_2', f'{id}.txt')

    img = cv2.imread(img_path)
    img3d = copy.deepcopy(img)
    lidar_points = read_points(lidar_path)
    calib_dict = read_calib(calib_path)
    annotation_dict = read_label(label_path)

    bboxes = annotation_dict['bbox']
    names = annotation_dict['name']
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    CLASSES = {
            'Pedestrian': 0, 
            'Cyclist': 1, 
            'Car': 2
            }


    ## 1. visualize 2d 
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[CLASSES.get(names[i], -1)], 2)
    cv2.imwrite(os.path.join(saved_root, f'{id}-2d.png'), img)
    cv2.imshow(f'{id}-2d bbox', img)
    cv2.waitKey(0)

    ## 2. visualize 3d bbox in point cloud

    # 2.1 camera coordinates
    dimensions = annotation_dict['dimensions']
    location = annotation_dict['location']
    rotation_y = annotation_dict['rotation_y']
    names = annotation_dict['name']

    # 2.2 lidar coordinates
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
    tr_velo_to_cam = calib_dict['Tr_velo_to_cam']
    r0_rect = calib_dict['R0_rect']
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
    lidar_bboxes_points = bbox3d2corners(bboxes_lidar) # (N, 8, 3)
    labels = [CLASSES.get(name, -1) for name in names]
    vis_pc(lidar_points, lidar_bboxes_points, labels) # (N, 8, 2)

    ## 3. visualize 3d bbox in image
    P2 = calib_dict['P2']
    image_points = points_lidar2image(lidar_bboxes_points, tr_velo_to_cam, r0_rect, P2)
    img3d = vis_img_3d(img3d, image_points, labels)
    cv2.imwrite(os.path.join(saved_root, f'{id}-3d.png'), img3d)
    cv2.imshow(f'{id}-3d bbox', img3d)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data information')
    parser.add_argument('--root', default='/home/lifa/data/KITTI/training')
    parser.add_argument('--id', default='000134')
    parser.add_argument('--saved_root', default='tmp')
    args = parser.parse_args()
    
    root = args.root
    id = args.id
    saved_root = args.saved_root
    os.makedirs(saved_root, exist_ok=True)
    vis_gt(root, id, saved_root)
