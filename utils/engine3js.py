import math
from operator import itemgetter

import cv2
import numpy as np
from pythreejs import *


class Engine3js:
    """
    The implementation of these interfaces depends on Pythreejs, so make
    sure you install and authorize the Jupyter Widgets plug-in correctly.

    Attributes:
        view_width, view_height:   Size of the view window.
        position:                  Position of the camera.
        lookAtPos:                 The point at which the camera looks at.
        axis_size:                 The size of axis.
        grid_length:               grid size, length == width.
        grid_num:                  grid number.
        grid:                      If use grid.
        axis:                      If use axis.
    """

    def __init__(
        self,
        view_width=455,
        view_height=256,
        position=[300, 100, 0],
        lookAtPos=[0, 0, 0],
        axis_size=300,
        grid_length=600,
        grid_num=20,
        grid=False,
        axis=False,
    ):
        self.view_width = view_width
        self.view_height = view_height
        self.position = position

        # set the camera
        self.cam = PerspectiveCamera(position=self.position, aspect=self.view_width / self.view_height)
        self.cam.lookAt(lookAtPos)

        # x,y,z axis
        self.axis = AxesHelper(axis_size)  # axes length

        # set grid size
        self.gridHelper = GridHelper(grid_length, grid_num)

        # set scene
        self.scene = Scene(
            children=[
                self.cam,
                DirectionalLight(position=[3, 5, 1], intensity=0.6),
                AmbientLight(intensity=0.5),
            ]
        )

        # add axis or grid
        if grid:
            self.scene.add(self.gridHelper)

        if axis:
            self.scene.add(self.axis)

        # render the objects in scene
        self.renderer = Renderer(
            camera=self.cam,
            scene=self.scene,
            controls=[OrbitControls(controlling=self.cam)],
            width=self.view_width,
            height=self.view_height,
        )
        # display(renderer4)

    def get_width(self):
        return self.view_width

    def plot(self):
        self.renderer.render(self.scene, self.cam)

    def scene_add(self, object):
        self.scene.add(object)

    def scene_remove(self, object):
        self.scene.remove(object)


class Geometry:
    """
    This is the geometry base class that defines buffer and material.
    """

    def __init__(self, name="geometry"):
        self.geometry = None
        self.material = None
        self.name = name

    def get_Name():
        return self.name


class Skeleton(Geometry):
    """
    This is the class for drawing human body poses.
    """

    def __init__(self, name="skeleton", lineWidth=3, body_edges=[]):
        super(Skeleton, self).__init__(name)
        self.material = LineBasicMaterial(vertexColors="VertexColors", linewidth=lineWidth)
        self.colorSet = BufferAttribute(
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            ),
            normalized=False,
        )
        # self.geometry.attributes["color"] = self.colorSet
        self.body_edges = body_edges

    def __call__(self, poses_3d):
        poses = []
        for pose_position_tmp in poses_3d:
            bones = []
            for edge in self.body_edges:
                # put pair of points as limbs
                bones.append(pose_position_tmp[edge[0]])
                bones.append(pose_position_tmp[edge[1]])

            bones = np.asarray(bones, dtype=np.float32)

            # You can find the api in https://github.com/jupyter-widgets/pythreejs

            self.geometry = BufferGeometry(
                attributes={
                    "position": BufferAttribute(bones, normalized=False),
                    # It defines limbs' color
                    "color": self.colorSet,
                }
            )

            pose = LineSegments(self.geometry, self.material)
            poses.append(pose)
            # self.geometry.close()
        return poses

    def plot(self, pose_points=None):
        return self.__call__(pose_points)


class Cloudpoint(Geometry):
    """
    This is the class for drawing cloud points.
    """

    def __init__(self, name="cloudpoint", points=[], point_size=5, line=None, points_color="blue"):
        super(Cloudpoint, self).__init__(name)
        self.material = PointsMaterial(size=point_size, color=points_color)
        self.points = points
        self.line = line

    def __call__(self, points_3d):
        self.geometry = BufferGeometry(
            attributes={
                "position": BufferAttribute(points_3d, normalized=False),
                # It defines points' vertices' color
                "color": BufferAttribute(
                    np.array(
                        [
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                        ],
                        dtype=np.float32,
                    ),
                    normalized=False,
                ),
            },
        )
        cloud_points = Points(self.geometry, self.material)

        if self.line is not None:
            g1 = BufferGeometry(
                attributes={
                    "position": BufferAttribute(line, normalized=False),
                    # It defines limbs' color
                    "color": BufferAttribute(
                        # Here you can set vertex colors, if you set the 'color' option = vertexes
                        np.array(
                            [
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                            ],
                            dtype=np.float32,
                        ),
                        normalized=False,
                    ),
                },
            )
            m1 = LineBasicMaterial(color="red", linewidth=3)
            facemesh = LineSegments(g1, m1)
            return [cloud_points, facemesh]

        return cloud_points


def Box_bounding(Geometry):
    def __init__(self, name="Box", lineWidth=3):
        super(Box_bounding, self).__init__(name)
        self.material = LineBasicMaterial(vertexColors="VertexColors", linewidth=lineWidth)
        self.edge = []

    def __call__(self, points=None):
        pass


class Pose:
    num_kpts = 18
    kpt_names = [
        "neck",
        "nose",
        "l_sho",
        "l_elb",
        "l_wri",
        "l_hip",
        "l_knee",
        "l_ank",
        "r_sho",
        "r_elb",
        "r_wri",
        "r_hip",
        "r_knee",
        "r_ank",
        "r_eye",
        "l_eye",
        "r_ear",
        "l_ear",
    ]
    sigmas = (
        np.array(
            [
                0.79,
                0.26,
                0.79,
                0.72,
                0.62,
                1.07,
                0.87,
                0.89,
                0.79,
                0.72,
                0.62,
                1.07,
                0.87,
                0.89,
                0.25,
                0.25,
                0.35,
                0.35,
            ],
            dtype=np.float32,
        )
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None
        self.translation_filter = [
            OneEuroFilter(freq=80, beta=0.01),
            OneEuroFilter(freq=80, beta=0.01),
            OneEuroFilter(freq=80, beta=0.01),
        ]

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def filter(self, translation):
        filtered_translation = []
        for coordinate_id in range(3):
            filtered_translation.append(self.translation_filter[coordinate_id](translation[coordinate_id]))
        return filtered_translation


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses_sorted_ids = list(range(len(current_poses)))
    current_poses_sorted_ids = sorted(
        current_poses_sorted_ids,
        key=lambda pose_id: current_poses[pose_id].confidence,
        reverse=True,
    )  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in current_poses_sorted_ids:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
        if best_matched_pose_id is not None:
            current_poses[current_pose_id].translation_filter = previous_poses[best_matched_id].translation_filter


AVG_PERSON_HEIGHT = 180

# pelvis (body center) is missing, id == 2
map_id_to_panoptic = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18]

limbs = [[18, 17, 1], [16, 15, 1], [5, 4, 3], [8, 7, 6], [11, 10, 9], [14, 13, 12]]


def get_root_relative_poses(inference_results):
    features, heatmap, paf_map = inference_results

    upsample_ratio = 4
    found_poses = extract_poses(heatmap[0:-1], paf_map, upsample_ratio)[0]
    # scale coordinates to features space
    found_poses[:, 0:-1:3] /= upsample_ratio
    found_poses[:, 1:-1:3] /= upsample_ratio

    poses_2d = []
    num_kpt_panoptic = 19
    num_kpt = 18
    for pose_id in range(found_poses.shape[0]):
        if found_poses[pose_id, 5] == -1:  # skip pose if is not found neck
            continue
        pose_2d = np.ones(num_kpt_panoptic * 3 + 1, dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if found_poses[pose_id, kpt_id * 3 + 2] != -1:
                x_2d, y_2d = found_poses[pose_id, kpt_id * 3 : kpt_id * 3 + 2]
                conf = found_poses[pose_id, kpt_id * 3 + 2]
                pose_2d[map_id_to_panoptic[kpt_id] * 3] = x_2d  # just repacking
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 1] = y_2d
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 2] = conf
        pose_2d[-1] = found_poses[pose_id, -1]
        poses_2d.append(pose_2d)

    keypoint_treshold = 0.1
    poses_3d = np.ones((len(poses_2d), num_kpt_panoptic * 4), dtype=np.float32) * -1
    for pose_id in range(len(poses_3d)):
        if poses_2d[pose_id][2] > keypoint_treshold:
            neck_2d = poses_2d[pose_id][:2].astype(int)
            # read all pose coordinates at neck location
            for kpt_id in range(num_kpt_panoptic):
                map_3d = features[kpt_id * 3 : (kpt_id + 1) * 3]
                poses_3d[pose_id][kpt_id * 4] = map_3d[0, neck_2d[1], neck_2d[0]] * AVG_PERSON_HEIGHT
                poses_3d[pose_id][kpt_id * 4 + 1] = map_3d[1, neck_2d[1], neck_2d[0]] * AVG_PERSON_HEIGHT
                poses_3d[pose_id][kpt_id * 4 + 2] = map_3d[2, neck_2d[1], neck_2d[0]] * AVG_PERSON_HEIGHT
                poses_3d[pose_id][kpt_id * 4 + 3] = poses_2d[pose_id][kpt_id * 3 + 2]

            # refine keypoints coordinates at corresponding limbs locations
            for limb in limbs:
                for kpt_id_from in limb:
                    if poses_2d[pose_id][kpt_id_from * 3 + 2] > keypoint_treshold:
                        for kpt_id_where in limb:
                            kpt_from_2d = poses_2d[pose_id][kpt_id_from * 3 : kpt_id_from * 3 + 2].astype(int)
                            map_3d = features[kpt_id_where * 3 : (kpt_id_where + 1) * 3]
                            poses_3d[pose_id][kpt_id_where * 4] = map_3d[0, kpt_from_2d[1], kpt_from_2d[0]] * AVG_PERSON_HEIGHT
                            poses_3d[pose_id][kpt_id_where * 4 + 1] = map_3d[1, kpt_from_2d[1], kpt_from_2d[0]] * AVG_PERSON_HEIGHT
                            poses_3d[pose_id][kpt_id_where * 4 + 2] = map_3d[2, kpt_from_2d[1], kpt_from_2d[0]] * AVG_PERSON_HEIGHT
                        break

    return poses_3d, np.array(poses_2d), features.shape


previous_poses_2d = []


def parse_poses(inference_results, input_scale, stride, fx, is_video=False):
    global previous_poses_2d
    poses_3d, poses_2d, features_shape = get_root_relative_poses(inference_results)
    poses_2d_scaled = []
    for pose_2d in poses_2d:
        num_kpt = (pose_2d.shape[0] - 1) // 3
        pose_2d_scaled = np.ones(pose_2d.shape[0], dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if pose_2d[kpt_id * 3 + 2] != -1:
                pose_2d_scaled[kpt_id * 3] = int(pose_2d[kpt_id * 3] * stride / input_scale)
                pose_2d_scaled[kpt_id * 3 + 1] = int(pose_2d[kpt_id * 3 + 1] * stride / input_scale)
                pose_2d_scaled[kpt_id * 3 + 2] = pose_2d[kpt_id * 3 + 2]
        pose_2d_scaled[-1] = pose_2d[-1]
        poses_2d_scaled.append(pose_2d_scaled)

    if is_video:  # track poses ids
        current_poses_2d = []
        for pose_id in range(len(poses_2d_scaled)):
            pose_keypoints = np.ones((Pose.num_kpts, 2), dtype=np.int32) * -1
            for kpt_id in range(Pose.num_kpts):
                if poses_2d_scaled[pose_id][kpt_id * 3 + 2] != -1.0:  # keypoint is found
                    pose_keypoints[kpt_id, 0] = int(poses_2d_scaled[pose_id][kpt_id * 3 + 0])
                    pose_keypoints[kpt_id, 1] = int(poses_2d_scaled[pose_id][kpt_id * 3 + 1])
            pose = Pose(pose_keypoints, poses_2d_scaled[pose_id][-1])
            current_poses_2d.append(pose)
        propagate_ids(previous_poses_2d, current_poses_2d)
        previous_poses_2d = current_poses_2d

    translated_poses_3d = []
    # translate poses
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_2d = poses_2d[pose_id][:-1].reshape((-1, 3)).transpose()
        num_valid = np.count_nonzero(pose_2d[2] != -1)
        pose_3d_valid = np.zeros((3, num_valid), dtype=np.float32)
        pose_2d_valid = np.zeros((2, num_valid), dtype=np.float32)
        valid_id = 0
        for kpt_id in range(pose_3d.shape[1]):
            if pose_2d[2, kpt_id] == -1:
                continue
            pose_3d_valid[:, valid_id] = pose_3d[0:3, kpt_id]
            pose_2d_valid[:, valid_id] = pose_2d[0:2, kpt_id]
            valid_id += 1

        pose_2d_valid[0] = pose_2d_valid[0] - features_shape[2] / 2
        pose_2d_valid[1] = pose_2d_valid[1] - features_shape[1] / 2
        mean_3d = np.expand_dims(pose_3d_valid.mean(axis=1), axis=1)
        mean_2d = np.expand_dims(pose_2d_valid.mean(axis=1), axis=1)
        numerator = np.trace(
            np.dot(
                (pose_3d_valid[:2, :] - mean_3d[:2, :]).transpose(),
                pose_3d_valid[:2, :] - mean_3d[:2, :],
            )
        ).sum()
        numerator = np.sqrt(numerator)
        denominator = np.sqrt(
            np.trace(
                np.dot(
                    (pose_2d_valid[:2, :] - mean_2d[:2, :]).transpose(),
                    pose_2d_valid[:2, :] - mean_2d[:2, :],
                )
            ).sum()
        )
        mean_2d = np.array([mean_2d[0, 0], mean_2d[1, 0], fx * input_scale / stride])
        mean_3d = np.array([mean_3d[0, 0], mean_3d[1, 0], 0])
        translation = numerator / denominator * mean_2d - mean_3d

        if is_video:
            translation = current_poses_2d[pose_id].filter(translation)
        for kpt_id in range(19):
            pose_3d[0, kpt_id] = pose_3d[0, kpt_id] + translation[0]
            pose_3d[1, kpt_id] = pose_3d[1, kpt_id] + translation[1]
            pose_3d[2, kpt_id] = pose_3d[2, kpt_id] + translation[2]
        translated_poses_3d.append(pose_3d.transpose().reshape(-1))

    return np.array(translated_poses_3d), np.array(poses_2d_scaled)


def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)


class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered


class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=1, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered


BODY_PARTS_KPT_IDS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]
BODY_PARTS_PAF_IDS = (
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
)


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_center = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 1 : heatmap_with_borders.shape[1] - 1]
    heatmap_left = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 2 : heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 0 : heatmap_with_borders.shape[1] - 2]
    heatmap_up = heatmap_with_borders[2 : heatmap_with_borders.shape[0], 1 : heatmap_with_borders.shape[1] - 1]
    heatmap_down = heatmap_with_borders[0 : heatmap_with_borders.shape[0] - 2, 1 : heatmap_with_borders.shape[1] - 1]

    heatmap_peaks = (heatmap_center > heatmap_left) & (heatmap_center > heatmap_right) & (heatmap_center > heatmap_up) & (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1 : heatmap_center.shape[0] - 1, 1 : heatmap_center.shape[1] - 1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (
            keypoints[i][0],
            keypoints[i][1],
            heatmap[keypoints[i][1], keypoints[i][0]],
            total_keypoint_num + keypoint_num,
        )
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1  # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]  # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (
                    int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                    int(round((kpt_a[1] + kpt_b[1]) * 0.5)),
                )
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = vec[0] * part_pafs[0, mid_point[0][1], mid_point[0][0]] + vec[1] * part_pafs[1, mid_point[1][1], mid_point[1][0]]

                height_n = pafs.shape[1] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                ratio = 0
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        px = int(x[point_idx])
                        py = int(y[point_idx])
                        paf = part_pafs[:, py, px]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


def extract_poses(heatmaps, pafs, upsample_ratio):
    heatmaps = np.transpose(heatmaps, (1, 2, 0))
    pafs = np.transpose(pafs, (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, dsize=None, fx=upsample_ratio, fy=upsample_ratio)
    pafs = cv2.resize(pafs, dsize=None, fx=upsample_ratio, fy=upsample_ratio)
    heatmaps = np.transpose(heatmaps, (2, 0, 1))
    pafs = np.transpose(pafs, (2, 0, 1))

    num_keypoints = heatmaps.shape[0]
    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(heatmaps[kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

    found_poses = []
    for pose_entry in pose_entries:
        if len(pose_entry) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints * 3 + 1), dtype=np.float32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entry[kpt_id] != -1.0:
                pose_keypoints[kpt_id * 3 + 0] = all_keypoints[int(pose_entry[kpt_id]), 0]
                pose_keypoints[kpt_id * 3 + 1] = all_keypoints[int(pose_entry[kpt_id]), 1]
                pose_keypoints[kpt_id * 3 + 2] = all_keypoints[int(pose_entry[kpt_id]), 2]
        pose_keypoints[-1] = pose_entry[18]
        found_poses.append(pose_keypoints)

    if not found_poses:
        return np.array(found_poses, dtype=np.float32).reshape((0, 0)), None

    return np.array(found_poses, dtype=np.float32), None
