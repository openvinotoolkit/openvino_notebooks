import copy
import numba
import numpy as np
import os
import pdb
from pointpillars.utils import bbox3d2bevcorners, box_collision_test, read_points, \
    remove_pts_in_bboxes, limit_period


def dbsample(CLASSES, data_root, data_dict, db_sampler, sample_groups):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    db_infos: dict(Pedestrian, Cyclist, Car, ...)
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
    gt_difficulty = data_dict['difficulty']
    image_info, calib_info = data_dict['image_info'], data_dict['calib_info']

    sampled_pts, sampled_names, sampled_labels = [], [], []
    sampled_bboxes, sampled_difficulty = [], []

    avoid_coll_boxes = copy.deepcopy(gt_bboxes_3d)
    for name, v in sample_groups.items():
        # 1. calculate sample numbers
        sampled_num = v - np.sum(gt_names == name)
        if sampled_num <= 0:
            continue

        # 2. sample databases bboxes
        sampled_cls_list = db_sampler[name].sample(sampled_num)
        sampled_cls_bboxes = np.array([item['box3d_lidar'] for item in sampled_cls_list], dtype=np.float32)

        # 3. box_collision_test
        avoid_coll_boxes_bv_corners = bbox3d2bevcorners(avoid_coll_boxes)
        sampled_cls_bboxes_bv_corners = bbox3d2bevcorners(sampled_cls_bboxes)
        coll_query_matrix = np.concatenate([avoid_coll_boxes_bv_corners, sampled_cls_bboxes_bv_corners], axis=0)
        coll_mat = box_collision_test(coll_query_matrix, coll_query_matrix)
        n_gt, tmp_bboxes = len(avoid_coll_boxes_bv_corners), []
        for i in range(n_gt, len(coll_mat)):
            if any(coll_mat[i]):
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                cur_sample = sampled_cls_list[i - n_gt]
                pt_path = os.path.join(data_root, cur_sample['path'])
                sampled_pts_cur = read_points(pt_path)
                sampled_pts_cur[:, :3] += cur_sample['box3d_lidar'][:3]
                sampled_pts.append(sampled_pts_cur)
                sampled_names.append(cur_sample['name'])
                sampled_labels.append(CLASSES[cur_sample['name']])
                sampled_bboxes.append(cur_sample['box3d_lidar'])
                tmp_bboxes.append(cur_sample['box3d_lidar'])
                sampled_difficulty.append(cur_sample['difficulty'])
        if len(tmp_bboxes) == 0:
            tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
        else:
            tmp_bboxes = np.array(tmp_bboxes)
        avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        
    # merge sampled database
    # remove raw points in sampled_bboxes firstly
    pts = remove_pts_in_bboxes(pts, np.stack(sampled_bboxes, axis=0))
    # pts = np.concatenate([pts, np.concatenate(sampled_pts, axis=0)], axis=0)
    pts = np.concatenate([np.concatenate(sampled_pts, axis=0), pts], axis=0)
    gt_bboxes_3d = avoid_coll_boxes.astype(np.float32)
    gt_labels = np.concatenate([gt_labels, np.array(sampled_labels)], axis=0)
    gt_names = np.concatenate([gt_names, np.array(sampled_names)], axis=0)
    difficulty = np.concatenate([gt_difficulty, np.array(sampled_difficulty)], axis=0)
    data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels, 
            'gt_names': gt_names,
            'difficulty': difficulty,
            'image_info': image_info,
            'calib_info': calib_info
        }
    return data_dict


@numba.jit(nopython=True)
def object_noise_core(pts, gt_bboxes_3d, bev_corners, trans_vec, rot_angle, rot_mat, masks):
    '''
    pts: (N, 4)
    gt_bboxes_3d: (n_bbox, 7)
    bev_corners: ((n_bbox, 4, 2))
    trans_vec: (n_bbox, num_try, 3)
    rot_mat: (n_bbox, num_try, 2, 2)
    masks: (N, n_bbox), bool
    return: gt_bboxes_3d, pts
    '''
    # 1. select the noise of num_try for each bbox under the collision test
    n_bbox, num_try = trans_vec.shape[:2]
    
    # succ_mask: (n_bbox, ), whether each bbox can be added noise successfully. -1 denotes failure.
    succ_mask = -np.ones((n_bbox, ), dtype=np.int_)
    for i in range(n_bbox):
        for j in range(num_try):
            cur_bbox = bev_corners[i] - np.expand_dims(gt_bboxes_3d[i, :2], 0) # (4, 2) - (1, 2) -> (4, 2)
            rot = np.zeros((2, 2), dtype=np.float32)
            rot[:] = rot_mat[i, j] # (2, 2)
            trans = trans_vec[i, j] # (3, )
            cur_bbox = cur_bbox @ rot
            cur_bbox += gt_bboxes_3d[i, :2]
            cur_bbox += np.expand_dims(trans[:2], 0) # (4, 2)
            coll_mat = box_collision_test(np.expand_dims(cur_bbox, 0), bev_corners)
            coll_mat[0, i] = False
            if coll_mat.any():
                continue
            else:
                bev_corners[i] = cur_bbox # update the bev_corners when adding noise succseefully.
                succ_mask[i] = j
                break
    # 2. points and bboxes noise
    visit = {}
    for i in range(n_bbox):
        jj = succ_mask[i] 
        if jj == -1:
            continue
        cur_trans, cur_angle = trans_vec[i, jj], rot_angle[i, jj]
        cur_rot_mat = np.zeros((2, 2), dtype=np.float32)
        cur_rot_mat[:] = rot_mat[i, jj]
        for k in range(len(pts)):
            if masks[k][i] and k not in visit:
                cur_pt = pts[k] # (4, )
                cur_pt_xyz = np.zeros((1, 3), dtype=np.float32)
                cur_pt_xyz[0] = cur_pt[:3] - gt_bboxes_3d[i][:3]
                tmp_cur_pt_xy = np.zeros((1, 2), dtype=np.float32)
                tmp_cur_pt_xy[:] = cur_pt_xyz[:, :2]
                cur_pt_xyz[:, :2] = tmp_cur_pt_xy @ cur_rot_mat # (1, 2)
                cur_pt_xyz[0] = cur_pt_xyz[0] + gt_bboxes_3d[i][:3]
                cur_pt_xyz[0] = cur_pt_xyz[0] + cur_trans[:3]
                cur_pt[:3] = cur_pt_xyz[0]
                visit[k] = 1

        gt_bboxes_3d[i, :3] += cur_trans[:3]
        gt_bboxes_3d[i, 6] += cur_angle

    return gt_bboxes_3d, pts


def object_noise(data_dict, num_try, translation_std, rot_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    num_try: int, 100
    translation_std: shape=[3, ]
    rot_range: shape=[2, ]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    n_bbox = len(gt_bboxes_3d)
    
    # 1. generate rotation vectors and rotation matrices
    trans_vec = np.random.normal(scale=translation_std, size=(n_bbox, num_try, 3)).astype(np.float32)
    rot_angle = np.random.uniform(rot_range[0], rot_range[1], size=(n_bbox, num_try)).astype(np.float32)
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n_bbox, num_try)
    rot_mat = np.transpose(rot_mat, (2, 3, 1, 0)) # (n_bbox, num_try, 2, 2)
    
    # 2. generate noise for each bbox and the points inside the bbox.
    bev_corners = bbox3d2bevcorners(gt_bboxes_3d) # (n_bbox, 4, 2) # for collision test
    masks = remove_pts_in_bboxes(pts, gt_bboxes_3d, rm=False) # identify which point should be added noise
    gt_bboxes_3d, pts = object_noise_core(pts=pts, 
                                          gt_bboxes_3d=gt_bboxes_3d, 
                                          bev_corners=bev_corners, 
                                          trans_vec=trans_vec, 
                                          rot_angle=rot_angle, 
                                          rot_mat=rot_mat, 
                                          masks=masks)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})

    return data_dict


def random_flip(data_dict, random_flip_ratio):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    random_flip_ratio: float, 0-1
    return: data_dict
    '''
    random_flip_state = np.random.choice([True, False], p=[random_flip_ratio, 1-random_flip_ratio])
    if random_flip_state:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        pts[:, 1] = -pts[:, 1] 
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi
        data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
        data_dict.update({'pts': pts})
    return data_dict


def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    rot_range: [a, b]
    scale_ratio_range: [c, d] 
    translation_std:  [e, f, g]
    return: data_dict
    '''
    pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
    
    # 1. rotation
    rot_angle = np.random.uniform(rot_range[0], rot_range[1])
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2)
    # 1.1 bbox rotation
    gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
    gt_bboxes_3d[:, 6] += rot_angle
    # 1.2 point rotation
    pts[:, :2] = pts[:, :2] @ rot_mat.T

    # 2. scaling
    scale_fator = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
    gt_bboxes_3d[:, :6] *= scale_fator
    pts[:, :3] *= scale_fator

    # 3. translation
    trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
    gt_bboxes_3d[:, :3] += trans_factor
    pts[:, :3] += trans_factor
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'pts': pts})
    return data_dict


def point_range_filter(data_dict, point_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    pts = data_dict['pts']
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    data_dict.update({'pts': pts})
    return data_dict 


def object_range_filter(data_dict, object_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']

    # bev filter
    flag_x_low = gt_bboxes_3d[:, 0] > object_range[0]
    flag_y_low = gt_bboxes_3d[:, 1] > object_range[1]
    flag_x_high = gt_bboxes_3d[:, 0] < object_range[3]
    flag_y_high = gt_bboxes_3d[:, 1] < object_range[4]
    keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high

    gt_bboxes_3d, gt_labels = gt_bboxes_3d[keep_mask], gt_labels[keep_mask]
    gt_names, difficulty = gt_names[keep_mask], difficulty[keep_mask]
    gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict


def points_shuffle(data_dict):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    '''
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    data_dict.update({'pts': pts})
    return data_dict


def filter_bboxes_with_labels(data_dict, label=-1):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    label: int
    '''
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']
    idx = gt_labels != label
    gt_bboxes_3d = gt_bboxes_3d[idx]
    gt_labels = gt_labels[idx]
    gt_names = gt_names[idx]
    difficulty = difficulty[idx]
    data_dict.update({'gt_bboxes_3d': gt_bboxes_3d})
    data_dict.update({'gt_labels': gt_labels})
    data_dict.update({'gt_names': gt_names})
    data_dict.update({'difficulty': difficulty})
    return data_dict


def data_augment(CLASSES, data_root, data_dict, data_aug_config):
    '''
    CLASSES: dict(Pedestrian=0, Cyclist=1, Car=2)
    data_root: str, data root
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    data_aug_config: dict()
    return: data_dict
    '''

    # 1. sample databases and merge into the data 
    db_sampler_config = data_aug_config['db_sampler']
    data_dict = dbsample(CLASSES,
                         data_root,
                         data_dict, 
                         db_sampler=db_sampler_config['db_sampler'],
                         sample_groups=db_sampler_config['sample_groups'])
    # 2. object noise
    object_noise_config = data_aug_config['object_noise']
    data_dict = object_noise(data_dict, 
                             num_try=object_noise_config['num_try'],
                             translation_std=object_noise_config['translation_std'],
                             rot_range=object_noise_config['rot_range'])
    
    # 3. random flip
    random_flip_ratio = data_aug_config['random_flip_ratio']
    data_dict = random_flip(data_dict, random_flip_ratio)

    # 4. global rotation, scaling and translation
    global_rot_scale_trans_config = data_aug_config['global_rot_scale_trans']
    rot_range = global_rot_scale_trans_config['rot_range']
    scale_ratio_range = global_rot_scale_trans_config['scale_ratio_range']
    translation_std = global_rot_scale_trans_config['translation_std']
    data_dict = global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std)

    # 5. points range filter
    point_range = data_aug_config['point_range_filter']
    data_dict = point_range_filter(data_dict, point_range)

    # 6. object range filter
    object_range = data_aug_config['object_range_filter']
    data_dict = object_range_filter(data_dict, object_range)

    # 7. points shuffle
    data_dict = points_shuffle(data_dict)

    # # 8. filter bboxes with label=-1
    # data_dict = filter_bboxes_with_labels(data_dict)
    
    return data_dict
