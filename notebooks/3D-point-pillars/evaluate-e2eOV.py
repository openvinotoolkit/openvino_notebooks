"""
Evaluate PointPillars using E2E OpenVINO inference
Adapted from evaluate.py
"""

import argparse
import numpy as np
import os
from tqdm import tqdm

import sys
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from pointpillars.utils import keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev
from pointpillars.dataset import Kitti, get_dataloader

from e2eOVInference import E2EOVInference


def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    """Calculate score thresholds for PR curve"""
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def do_eval(det_results, gt_results, CLASSES, saved_path):
    """Evaluate detection results against ground truth"""

    # Import torch only for metric computation
    import torch

    # Device for metric computation
    device = torch.device('cpu')

    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    # 1. calculate iou
    ious = {
        'bbox_2d': [],
        'bbox_bev': [],
        'bbox_3d': []
    }
    ids = list(sorted(gt_results.keys()))
    for id in ids:
        gt_result = gt_results[id]['annos']
        det_result = det_results[id]

        # 1.1, 2d bboxes iou
        gt_bboxes2d = gt_result['bbox'].astype(np.float32)
        det_bboxes2d = det_result['bbox'].astype(np.float32)
        gt_bboxes2d_t = torch.from_numpy(gt_bboxes2d).to(device)
        det_bboxes2d_t = torch.from_numpy(det_bboxes2d).to(device)
        iou2d_v = iou2d(gt_bboxes2d_t, det_bboxes2d_t)
        ious['bbox_2d'].append(iou2d_v.cpu().numpy())

        # 1.2, bev iou
        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)

        gt_bev = np.concatenate([gt_location[:, [0, 2]], gt_dimensions[:, [0, 2]], gt_rotation_y[:, None]], axis=-1)
        det_bev = np.concatenate([det_location[:, [0, 2]], det_dimensions[:, [0, 2]], det_rotation_y[:, None]], axis=-1)
        gt_bev_t = torch.from_numpy(gt_bev).to(device)
        det_bev_t = torch.from_numpy(det_bev).to(device)
        iou_bev_v = iou_bev(gt_bev_t, det_bev_t)
        ious['bbox_bev'].append(iou_bev_v.cpu().numpy())

        # 1.3, 3dbboxes iou
        gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        gt_bboxes3d_t = torch.from_numpy(gt_bboxes3d).to(device)
        det_bboxes3d_t = torch.from_numpy(det_bboxes3d).to(device)
        iou3d_v = iou3d_camera(gt_bboxes3d_t, det_bboxes3d_t)
        ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    MIN_IOUS = {
        'Pedestrian': [0.5, 0.5, 0.5],
        'Cyclist': [0.5, 0.5, 0.5],
        'Car': [0.7, 0.7, 0.7]
    }
    MIN_HEIGHT = [40, 25, 25]

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_2d', 'bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]
            for difficulty in [0, 1, 2]:
                # Calculate metrics (simplified version - core logic same as evaluate-multibackend.py)
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                total_gt_alpha, total_det_alpha = [], []

                for id in ids:
                    gt_result = gt_results[id]['annos']
                    det_result = det_results[id]

                    # GT bbox property
                    cur_gt_names = gt_result['name']
                    cur_difficulty = gt_result['difficulty']
                    gt_ignores, dc_bboxes = [], []
                    for j, cur_gt_name in enumerate(cur_gt_names):
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                            valid_class = 0
                        elif cls == 'Car' and cur_gt_name == 'Van':
                            valid_class = 0
                        else:
                            valid_class = -1

                        if valid_class == 1 and not ignore:
                            gt_ignores.append(0)
                        elif valid_class == 0 or (valid_class == 1 and ignore):
                            gt_ignores.append(1)
                        else:
                            gt_ignores.append(-1)

                        if cur_gt_name == 'DontCare':
                            dc_bboxes.append(gt_result['bbox'][j])
                    total_gt_ignores.append(gt_ignores)
                    total_dc_bboxes.append(np.array(dc_bboxes))
                    total_gt_alpha.append(gt_result['alpha'])

                    # Det bbox property
                    cur_det_names = det_result['name']
                    cur_det_heights = det_result['bbox'][:, 3] - det_result['bbox'][:, 1]
                    det_ignores = []
                    for j, cur_det_name in enumerate(cur_det_names):
                        if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                            det_ignores.append(1)
                        elif cur_det_name == cls:
                            det_ignores.append(0)
                        else:
                            det_ignores.append(-1)
                    total_det_ignores.append(det_ignores)
                    total_scores.append(det_result['score'])
                    total_det_alpha.append(det_result['alpha'])

                # Calculate TP scores
                tp_scores = []
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)

                total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)

                # Calculate precision and recall
                tps, fns, fps, total_aos = [], [], [], []
                for score_threshold in score_thresholds:
                    tp, fn, fp, aos = 0, 0, 0, 0
                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm, ), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
                                    if cur_eval_ious[j, k] > match_iou:
                                        match_id = k
                                        match_iou = cur_eval_ious[j, k]

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1
                                    if eval_type == 'bbox_2d':
                                        aos += (1 + np.cos(gt_alpha[j] - det_alpha[match_id])) / 2
                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1

                        for k in range(mm):
                            if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                fp += 1

                        # Handle DontCare bboxes for 2D evaluation
                        if eval_type == 'bbox_2d':
                            dc_bboxes = total_dc_bboxes[i]
                            det_bboxes = det_results[id]['bbox']
                            if len(dc_bboxes) > 0:
                                ious_dc = iou2d(torch.from_numpy(det_bboxes).to(device),
                                               torch.from_numpy(dc_bboxes).to(device)).cpu().numpy()
                                for k in range(mm):
                                    if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                        if ious_dc[k].max() > CLS_MIN_IOU:
                                            fp -= 1

                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == 'bbox_2d':
                        total_aos.append(aos)

                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

                recalls = tps / (tps + fns)
                precisions = tps / (tps + fps)
                for i in range(len(score_thresholds)):
                    precisions[i] = np.max(precisions[i:])

                sums_AP = 0
                for i in range(0, len(score_thresholds), 4):
                    sums_AP += precisions[i]
                mAP = sums_AP / 11 * 100
                eval_ap_results[cls].append(mAP)

                if eval_type == 'bbox_2d':
                    total_aos = np.array(total_aos)
                    similarity = total_aos / (tps + fps)
                    for i in range(len(score_thresholds)):
                        similarity[i] = np.max(similarity[i:])
                    sums_similarity = 0
                    for i in range(0, len(score_thresholds), 4):
                        sums_similarity += similarity[i]
                    mSimilarity = sums_similarity / 11 * 100
                    eval_aos_results[cls].append(mSimilarity)

        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        if eval_type == 'bbox_2d':
            print(f'==========AOS==========')
            print(f'==========AOS==========', file=f)
            for k, v in eval_aos_results.items():
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
        if eval_type == 'bbox_2d':
            overall_results['AOS'] = np.mean(list(eval_aos_results.values()), 0)

    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
    f.close()


def main(args):
    """Main evaluation function"""
    # Allow selecting which point file extension / folder to use via CLI
    val_dataset = Kitti(data_root=args.data_root,
                        split=args.split,
                        pts_prefix=args.pts_prefix,
                        max_samples=args.nsamples if args.nsamples > 0 else None)

    # If requested, switch dataset to use .pcd files instead of .bin. This assumes
    # corresponding PCD files exist under a sibling folder (e.g. velodyne_reduced_pcd).
    if args.use_pcd:
        # adjust pts_prefix to point to pcd sibling folder
        val_dataset.pts_prefix = args.pts_prefix + '_pcd'
        # update stored velodyne_path entries to refer to .pcd extension
        for k, info in val_dataset.data_infos.items():
            vp = info.get('velodyne_path')
            if vp and vp.endswith('.bin'):
                info['velodyne_path'] = vp[:-4] + '.pcd'

    val_dataloader = get_dataloader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}

    # Initialize E2E OpenVINO inference engine
    print("Initializing E2E OpenVINO inference engine...")
    inference_engine = E2EOVInference(
        config_path=args.config,
        device=args.device.upper()
    )

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    format_results = {}
    print('Predicting the results.')
    for i, data_dict in enumerate(tqdm(val_dataloader)):
        batched_pts = data_dict['batched_pts']

        # Process each sample in batch
        batch_results = []
        for j, pts in enumerate(batched_pts):
            # Run inference
            res = inference_engine.infer(pts)
            batch_results.append(res)

        # Format results
        for j, result in enumerate(batch_results):
            format_result = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }

            calib_info = data_dict['batched_calib_info'][j]
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)
            P2 = calib_info['P2'].astype(np.float32)
            image_shape = data_dict['batched_img_info'][j]['image_shape']
            idx = data_dict['batched_img_info'][j]['image_idx']

            result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']

            for lidar_bbox, label, score, bbox2d, camera_bbox in \
                zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                format_result['name'].append(LABEL2CLASSES[label])
                format_result['truncated'].append(0.0)
                format_result['occluded'].append(0)
                alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                format_result['alpha'].append(alpha)
                format_result['bbox'].append(bbox2d)
                format_result['dimensions'].append(camera_bbox[3:6])
                format_result['location'].append(camera_bbox[:3])
                format_result['rotation_y'].append(camera_bbox[6])
                format_result['score'].append(score)

            write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))
            format_results[idx] = {k: np.array(v) for k, v in format_result.items()}

    write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))

    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)


if __name__ == '__main__':
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='E2E OpenVINO PointPillars Evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', default=f'{current_file_path}/Datasets',
                        help='Data root for KITTI dataset')
    parser.add_argument('--config', default=f'{current_file_path}/pretrained/pointpillars_ov_config.json',
                        help='Path to OpenVINO config JSON')
    parser.add_argument('--saved_path', default=f'{current_file_path}/Datasets/results_e2e_ov',
                        help='Path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--nsamples', type=int, default=0,
                        help='Number of samples to evaluate (0 for all)')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU'],
                        help='OpenVINO device')
    parser.add_argument('--pts-prefix', default='velodyne_reduced',
                        help='Pointcloud folder prefix to use (mirrors Kitti default)')
    parser.add_argument('--use-pcd', action='store_true',
                        help='Load .pcd files instead of .bin (expects xxx_pcd sibling folder)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'trainval', 'test'],
                        help='Which KITTI split to evaluate')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("E2E OpenVINO PointPillars Evaluation")
    print("="*60)
    print("\nConfiguration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    main(args)
