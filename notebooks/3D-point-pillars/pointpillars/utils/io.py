import numpy as np
import os
import pickle

_read_kitti_pcd = None

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply', '.pcd']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    elif suffix == '.pcd':
        # import here to avoid circular imports: pointpillars.utils <-> pointpillars.dataset
        global _read_kitti_pcd
        if _read_kitti_pcd is None:
            from pointpillars.dataset.kitti_open3d.utils import read_kitti_pcd as _impl
            _read_kitti_pcd = _impl
        pts = _read_kitti_pcd(file_path)

        return pts.astype(np.float32)
    else:
        # .ply not implemented yet
        raise NotImplementedError


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float32)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float32)

    return annotation


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
        result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
        result['score']

    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)
