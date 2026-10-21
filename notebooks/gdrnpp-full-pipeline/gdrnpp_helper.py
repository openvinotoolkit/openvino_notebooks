"""Inference-only GDRNPP building blocks used by gdrnpp-full-pipeline.ipynb.

The YOLOX detector and GDRN pose network are re-implemented in plain PyTorch with
parameter names matching the released GDRNPP checkpoints
(https://github.com/shanice-l/gdrnpp_bop2022, Apache-2.0; YOLOX (c) Megvii, Apache-2.0).
Pre- and post-processing use only NumPy and OpenCV.
"""

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# YOLOX-X detector
# ---------------------------------------------------------------------------


class BaseConv(nn.Module):
    def __init__(self, cin, cout, ksize, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, ksize, stride, (ksize - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut):
        super().__init__()
        self.conv1 = BaseConv(channels, channels, 1)
        self.conv2 = BaseConv(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n, shortcut=True):
        super().__init__()
        hidden = cout // 2
        self.conv1 = BaseConv(cin, hidden, 1)
        self.conv2 = BaseConv(cin, hidden, 1)
        self.conv3 = BaseConv(2 * hidden, cout, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class Focus(nn.Module):
    def __init__(self, cout):
        super().__init__()
        self.conv = BaseConv(12, cout, 3)

    def forward(self, x):
        tl, tr = x[..., ::2, ::2], x[..., ::2, 1::2]
        bl, br = x[..., 1::2, ::2], x[..., 1::2, 1::2]
        return self.conv(torch.cat((tl, bl, tr, br), dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden = channels // 2
        self.conv1 = BaseConv(channels, hidden, 1)
        self.m = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes)
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.m], dim=1))


class CSPDarknet(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.stem = Focus(c)
        self.dark2 = nn.Sequential(BaseConv(c, 2 * c, 3, 2), CSPLayer(2 * c, 2 * c, n))
        self.dark3 = nn.Sequential(BaseConv(2 * c, 4 * c, 3, 2), CSPLayer(4 * c, 4 * c, 3 * n))
        self.dark4 = nn.Sequential(BaseConv(4 * c, 8 * c, 3, 2), CSPLayer(8 * c, 8 * c, 3 * n))
        self.dark5 = nn.Sequential(BaseConv(8 * c, 16 * c, 3, 2), SPPBottleneck(16 * c), CSPLayer(16 * c, 16 * c, n, shortcut=False))

    def forward(self, x):
        x3 = self.dark3(self.dark2(self.stem(x)))
        x4 = self.dark4(x3)
        return x3, x4, self.dark5(x4)


class YOLOPAFPN(nn.Module):
    def __init__(self, c=80, n=4):
        super().__init__()
        c0, c1, c2 = 4 * c, 8 * c, 16 * c
        self.backbone = CSPDarknet(c, n)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(c2, c1, 1)
        self.C3_p4 = CSPLayer(2 * c1, c1, n, False)
        self.reduce_conv1 = BaseConv(c1, c0, 1)
        self.C3_p3 = CSPLayer(2 * c0, c0, n, False)
        self.bu_conv2 = BaseConv(c0, c0, 3, 2)
        self.C3_n3 = CSPLayer(2 * c0, c1, n, False)
        self.bu_conv1 = BaseConv(c1, c1, 3, 2)
        self.C3_n4 = CSPLayer(2 * c1, c2, n, False)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, c=80):
        super().__init__()
        hidden = 4 * c

        def branch():
            return nn.Sequential(BaseConv(hidden, hidden, 3), BaseConv(hidden, hidden, 3))

        self.stems = nn.ModuleList(BaseConv(ch, hidden, 1) for ch in (4 * c, 8 * c, 16 * c))
        self.cls_convs = nn.ModuleList(branch() for _ in range(3))
        self.reg_convs = nn.ModuleList(branch() for _ in range(3))
        self.cls_preds = nn.ModuleList(nn.Conv2d(hidden, num_classes, 1) for _ in range(3))
        self.reg_preds = nn.ModuleList(nn.Conv2d(hidden, 4, 1) for _ in range(3))
        self.obj_preds = nn.ModuleList(nn.Conv2d(hidden, 1, 1) for _ in range(3))

    def forward(self, features):
        outputs = []
        for k, x in enumerate(features):
            x = self.stems[k](x)
            cls_out = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)
            out = torch.cat([self.reg_preds[k](reg_feat), self.obj_preds[k](reg_feat).sigmoid(), cls_out.sigmoid()], 1)
            outputs.append(out.flatten(start_dim=2))
        # [batch, anchors, 4 box + 1 objectness + num_classes], boxes not yet grid-decoded
        return torch.cat(outputs, dim=2).permute(0, 2, 1)


class YOLOX(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = YOLOPAFPN()
        self.head = YOLOXHead(num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_yolox(checkpoint_path: str | Path) -> YOLOX:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = YOLOX(num_classes=state["head.cls_preds.0.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval()


# ---------------------------------------------------------------------------
# GDRN pose network: ConvNeXt-B backbone + geometry head + Patch-PnP head
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(OrderedDict(fc1=nn.Linear(dim, 4 * dim), act=nn.GELU(), fc2=nn.Linear(4 * dim, dim)))
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = self.mlp(self.norm(self.conv_dw(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + y if self.gamma is None else x + y * self.gamma.reshape(1, -1, 1, 1)


class ConvNeXtStage(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        if cin != cout:
            self.downsample = nn.Sequential(LayerNorm2d(cin, eps=1e-6), nn.Conv2d(cin, cout, 2, 2))
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(cout) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class ConvNeXtBase(nn.Module):
    """ConvNeXt-B returning the stride-32 feature map."""

    def __init__(self, dims=(128, 256, 512, 1024), depths=(3, 3, 27, 3)):
        super().__init__()
        self.stem_0 = nn.Conv2d(3, dims[0], 4, 4)
        self.stem_1 = LayerNorm2d(dims[0], eps=1e-6)
        for i, depth in enumerate(depths):
            setattr(self, f"stages_{i}", ConvNeXtStage(dims[max(i - 1, 0)], dims[i], depth))

    def forward(self, x):
        x = self.stem_1(self.stem_0(x))
        return self.stages_3(self.stages_2(self.stages_1(self.stages_0(x))))


class ConvModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(32, channels)
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(self.norm(self.conv(x)))


class GeoHead(nn.Module):
    """Upsamples stride-32 features to 64x64 maps of masks, XYZ coordinates and surface regions."""

    def __init__(self, num_classes, in_dim=1024, feat_dim=256, region_dim=65):
        super().__init__()
        up = lambda: nn.UpsamplingBilinear2d(scale_factor=2)  # noqa: E731
        self.features = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_dim, feat_dim, 3, 2, padding=1, output_padding=1, bias=False),
                nn.GroupNorm(32, feat_dim),
                nn.GELU(),
                ConvModule(feat_dim),
                ConvModule(feat_dim),
                up(),
                ConvModule(feat_dim),
                ConvModule(feat_dim),
                up(),
                ConvModule(feat_dim),
                ConvModule(feat_dim),
            ]
        )
        self.out_layer = nn.Conv2d(feat_dim, (2 + 3 + region_dim) * num_classes, 1)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.out_layer(x)


class PnPNet(nn.Module):
    """Regresses allocentric 6D rotation and scale-invariant translation from dense correspondences."""

    def __init__(self, in_dim=69, feat_dim=128):
        super().__init__()
        layers = []
        for i in range(3):
            layers += [
                nn.Conv2d(in_dim if i == 0 else feat_dim, feat_dim, 3, 2, 1, bias=False),
                nn.GroupNorm(32, feat_dim),
                nn.GELU(),
            ]
        self.features = nn.Sequential(*layers)
        self.fc1 = nn.Linear(feat_dim * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, 6)
        self.fc_t = nn.Linear(256, 3)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc2(self.act(self.fc1(self.features(x).flatten(1)))))
        return self.fc_r(x), self.fc_t(x)


class GDRN(nn.Module):
    def __init__(self, num_classes, region_dim=65):
        super().__init__()
        self.num_classes = num_classes
        self.region_dim = region_dim
        self.backbone = ConvNeXtBase()
        self.geo_head_net = GeoHead(num_classes, region_dim=region_dim)
        self.pnp_net = PnPNet()

    def forward(self, image, roi_classes, roi_coord_2d, roi_extents):
        out = self.geo_head_net(self.backbone(image))
        b, _, h, w = out.shape
        c = self.num_classes
        rois = torch.arange(b)

        def select(maps, dim):
            return maps.reshape(b, c, dim, h, w)[rois, roi_classes]

        vis_mask = select(out[:, :c], 1)
        coor_x = select(out[:, 2 * c : 3 * c], 1)
        coor_y = select(out[:, 3 * c : 4 * c], 1)
        coor_z = select(out[:, 4 * c : 5 * c], 1)
        region = select(out[:, 5 * c :], self.region_dim)

        xyz = (torch.cat((coor_x, coor_y, coor_z), dim=1) - 0.5) * roi_extents.reshape(b, 3, 1, 1)
        pnp_input = torch.cat((xyz, roi_coord_2d, F.softmax(region[:, 1:], dim=1)), dim=1)
        pred_rot, pred_t = self.pnp_net(pnp_input)
        return pred_rot, pred_t, coor_x, coor_y, coor_z, vis_mask


def load_gdrn(checkpoint_path: str | Path) -> GDRN:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # ConvModule registers its GroupNorm as both `norm` and `gn`; the duplicates are identical.
    state = {k: v for k, v in state.items() if ".gn." not in k}
    model = GDRN(num_classes=state["geo_head_net.out_layer.weight"].shape[0] // 70)
    model.load_state_dict(state)
    return model.eval()


def _fold_layer_norm(norm: nn.LayerNorm, layer: nn.Module) -> None:
    """Fold the LayerNorm scale and shift into the next Linear or Conv2d layer: W' = W * scale, b' = b + W @ shift."""
    if norm.weight is None:
        return
    scale, shift = norm.weight.detach(), norm.bias.detach()
    weight = layer.weight.detach()
    shape = (1, -1) + (1,) * (weight.dim() - 2)
    bias = layer.bias.detach() + (weight * shift.reshape(shape)).flatten(1).sum(1)
    layer.weight.copy_(weight * scale.reshape(shape))
    layer.bias.copy_(bias)
    norm.weight, norm.bias, norm.elementwise_affine = None, None, False


@torch.no_grad()
def fold_norm_affine(model: GDRN) -> GDRN:
    """
    Fold ConvNeXt LayerNorm affine parameters and layer-scale constants into the adjacent layers.

    Removes the elementwise multiply/add operations from the converted graph without changing the outputs.
    """
    for i in range(4):
        stage = getattr(model.backbone, f"stages_{i}")
        if isinstance(stage.downsample, nn.Sequential):
            _fold_layer_norm(stage.downsample[0], stage.downsample[1])
        for block in stage.blocks:
            _fold_layer_norm(block.norm, block.mlp.fc1)
            if block.gamma is not None:
                gamma = block.gamma.detach()
                block.mlp.fc2.weight.mul_(gamma[:, None])
                block.mlp.fc2.bias.mul_(gamma)
                block.gamma = None
    return model


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def yolox_preprocess(image: np.ndarray, input_size: tuple = (640, 640)) -> tuple[np.ndarray, float]:
    """Letterbox a BGR image into a [1, 3, H, W] float32 tensor padded with 114."""
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    h, w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    padded = np.full((*input_size, 3), 114, dtype=np.uint8)
    padded[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def yolox_postprocess(
    raw: np.ndarray, input_size: tuple = (640, 640), conf_thr: float = 0.01, nms_thr: float = 0.65, strides: tuple = (8, 16, 32)
) -> np.ndarray:
    """Decode raw YOLOX output and apply class-agnostic NMS.

    Returns an (N, 7) array: x1, y1, x2, y2, objectness, class probability, class id.
    """
    grids, grid_strides = [], []
    for s in strides:
        yv, xv = np.meshgrid(np.arange(input_size[0] // s), np.arange(input_size[1] // s), indexing="ij")
        grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
        grid_strides.append(np.full((grids[-1].shape[0], 1), s))
    grids, grid_strides = np.concatenate(grids).astype(np.float32), np.concatenate(grid_strides).astype(np.float32)

    pred = raw[0].astype(np.float32)
    xy = (pred[:, :2] + grids) * grid_strides
    wh = np.exp(pred[:, 2:4]) * grid_strides
    boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)
    cls_ids = pred[:, 5:].argmax(1)
    cls_conf = pred[np.arange(len(pred)), 5 + cls_ids]
    scores = pred[:, 4] * cls_conf
    keep = scores >= conf_thr
    if not keep.any():
        return np.zeros((0, 7), dtype=np.float32)
    boxes, obj, cls_conf, cls_ids, scores = boxes[keep], pred[keep, 4], cls_conf[keep], cls_ids[keep], scores[keep]
    idx = np.array(_nms(boxes, scores, nms_thr))
    return np.column_stack([boxes[idx], obj[idx], cls_conf[idx], cls_ids[idx].astype(np.float32)])


def _crop_resize(image, center, scale, size):
    half = scale / 2
    src = np.float32([center, center + [0, -half], center + [-half, -half]])
    dst = np.float32([[size / 2, size / 2], [size / 2, 0], [0, 0]])
    return cv2.warpAffine(image, cv2.getAffineTransform(src, dst), (size, size), flags=cv2.INTER_LINEAR)


def prepare_rois(
    image: np.ndarray, detections: np.ndarray, ratio: float, extents: dict, input_res: int = 256, output_res: int = 64, pad_scale: float = 1.5
) -> tuple[dict, list]:
    """Crop each detection into GDRN inputs and keep the metadata needed to recover poses."""
    im_h, im_w = image.shape[:2]
    image_f = image.astype(np.float32)
    x = np.linspace(0, 1, im_w, dtype=np.float32, endpoint=False)
    y = np.linspace(0, 1, im_h, dtype=np.float32, endpoint=False)
    coord_2d = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0)

    inputs = {"image": [], "roi_classes": [], "roi_coord_2d": [], "roi_extents": []}
    meta = []
    for det in detections:
        x1, y1, x2, y2 = det[:4] / ratio
        class_id = int(det[6])
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        scale = min(max(bw, bh) * pad_scale, max(im_h, im_w))
        roi_coord = _crop_resize(coord_2d, center, scale, output_res).transpose(2, 0, 1)
        inputs["image"].append(_crop_resize(image_f, center, scale, input_res).transpose(2, 0, 1) / 255.0)
        inputs["roi_classes"].append(class_id)
        inputs["roi_coord_2d"].append(roi_coord)
        inputs["roi_extents"].append(extents[class_id])
        meta.append(
            dict(
                class_id=class_id,
                score=float(det[4] * det[5]),
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                center=center,
                roi_wh=np.array([bw, bh], dtype=np.float32),
                resize_ratio=output_res / scale,
                roi_coord_2d=roi_coord,
                extent=extents[class_id],
                im_shape=(im_h, im_w),
            )
        )
    dtypes = {"image": np.float32, "roi_classes": np.int64, "roi_coord_2d": np.float32, "roi_extents": np.float32}
    return {k: np.asarray(v, dtype=dtypes[k]) for k, v in inputs.items()}, meta


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _rot6d_to_mat(d6):
    x = d6[:3] / (np.linalg.norm(d6[:3]) + 1e-12)
    z = np.cross(x, d6[3:])
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.stack((x, np.cross(z, x), z), axis=-1)


def _allocentric_to_egocentric(rot, trans):
    obj_ray = trans / np.linalg.norm(trans)
    angle = np.arccos(obj_ray[2])
    if angle <= 0:
        return rot
    rot_vec = np.cross([0.0, 0.0, 1.0], obj_ray)
    rot_vec = rot_vec / np.linalg.norm(rot_vec) * angle
    return (cv2.Rodrigues(rot_vec)[0] @ rot).astype(np.float32)


def _direct_pose(pred_rot, pred_t, roi, cam_K):
    """Pose from the network's rotation/translation heads (fallback when PnP fails)."""
    cx = pred_t[0] * roi["roi_wh"][0] + roi["center"][0]
    cy = pred_t[1] * roi["roi_wh"][1] + roi["center"][1]
    z = pred_t[2] * roi["resize_ratio"]
    t = np.array([z * (cx - cam_K[0, 2]) / cam_K[0, 0], z * (cy - cam_K[1, 2]) / cam_K[1, 1], z], dtype=np.float32)
    return _allocentric_to_egocentric(_rot6d_to_mat(pred_rot), t), t


def _pnp_pose(xyz, mask, roi, cam_K, mask_thr):
    """EPnP + RANSAC on the 2D-3D correspondences inside the predicted visible mask."""
    im_h, im_w = roi["im_shape"]
    xyz = (xyz.transpose(1, 2, 0) - 0.5) * roi["extent"]
    coord = roi["roi_coord_2d"].transpose(1, 2, 0) * [im_w, im_h]
    keep = (mask > mask_thr) & np.all(np.abs(xyz) > 1e-4 * roi["extent"], axis=-1)
    if keep.sum() < 4:
        return None
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        xyz[keep].reshape(1, -1, 3).astype(np.float64),
        coord[keep].reshape(1, -1, 2).astype(np.float64),
        cam_K.astype(np.float64),
        np.zeros((8, 1)),
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=3.0,
        iterationsCount=100,
    )
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0].astype(np.float32), tvec.reshape(3).astype(np.float32)


def recover_poses(outputs: list, meta: list, cam_K: np.ndarray, mask_thr: float = 0.5) -> list:
    """Convert GDRN outputs into camera-frame poses (R, t in meters)."""
    pred_rot, pred_t, coor_x, coor_y, coor_z, vis_mask = outputs
    xyz = np.concatenate((coor_x, coor_y, coor_z), axis=1)
    flat = vis_mask.reshape(len(vis_mask), -1)
    lo, hi = flat.min(1), flat.max(1)
    masks = (vis_mask[:, 0] - lo[:, None, None]) / (hi - lo)[:, None, None]
    cv2.setRNGSeed(0)
    poses = []
    for i, roi in enumerate(meta):
        pose, method = _pnp_pose(xyz[i], masks[i], roi, cam_K, mask_thr), "pnp_ransac"
        if pose is None:
            pose, method = _direct_pose(pred_rot[i], pred_t[i], roi, cam_K), "direct"
        poses.append(dict(R=pose[0], t=pose[1], method=method, class_id=roi["class_id"], score=roi["score"], bbox=roi["bbox"]))
    return poses


def rotation_error_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r1.T @ r2) - 1) / 2, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_poses(image: np.ndarray, poses: list, extents: dict, cam_K: np.ndarray, names: list) -> np.ndarray:
    """Draw each pose as a projected 3D bounding box. Returns an RGB image."""
    canvas = image.copy()
    for pose in poses:
        color = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[pose["class_id"] * 32]]), cv2.COLORMAP_HSV)[0, 0])
        half = extents[pose["class_id"]] / 2
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        rvec = cv2.Rodrigues(pose["R"].astype(np.float64))[0]
        pts, _ = cv2.projectPoints(corners, rvec, pose["t"].astype(np.float64), cam_K.astype(np.float64), None)
        pts = pts.reshape(-1, 2).astype(int)
        for a, b in _EDGES:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        x1, y1 = pose["bbox"][:2].astype(int)
        label = f"{names[pose['class_id']]} {pose['score']:.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
