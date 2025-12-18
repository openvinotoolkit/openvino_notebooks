- 难点: 碰撞检测
- 代码难点: nms

## Datasets

- database 随机采样增强
  - 核心: 碰撞检测, 点是否在立方体内 
  - 输入: gt_bboxes_3d, pts, gt_labels
  - 输出: gt_bboxes_3d, pts, gt_labels
  - 逻辑: 
    1. 从Car, Pedestrian, Cyclist的database数据集中随机采集一定数量的bbox及inside points, 使每类bboxes的数量分别达到15, 10, 10.
    2. 将这些采样的bboxes进行碰撞检测, 满足碰撞检测的bboxes和对应labels加到gt_bboxes_3d, gt_labels
    3. 把位于这些采样bboxes内点删除掉, 替换成bboxes内部的点.
- object 随机旋转平移
  - 核心: 碰撞检测, 点是否在立方体内
  - 输入: gt_bboxes_3d, pts
  - 输出: gt_bboxes_3d, pts
  - 逻辑: 
    1. 以某个bbox为例, 随机产生num_try个平移向量t和旋转角度r, 旋转角度可以转成旋转矩阵(mat). 
    2. 对bbox进行旋转和平移, 找到num_try中第一个通过碰撞测试的平移向量t和旋转角度r(mat).
    3. 对bbox内部的点进行旋转和平移.
    4. 对bbox进行旋转和平移.
- 整体随机水平翻转
  - object: 水平翻转 (注意角度)
  - point: 水平翻转
- 整体旋转, 缩放和平移
  - object: 旋转, 缩放和平移, object的旋转需要注意, (x, y, z)和angle都需要变化.
  - point: 旋转, 缩放和平移
- 删除范围外的point
- 删除范围外的bbox
  - 需要注意的是: 这里做了旋转角度的归一化 (-pi, pi)
- point shuffle
- dataloader实现, `目前为了调试shuffle设置成了False, 之后需要设置成True`

## Model

- Pillar
  - cuda拓展
    ```
    cd ops
    python setup.py develop
    ```
  - Voxelization类实现, 这里基本是从mmdet3d复制过来的; 修改了`coors_out`的顺序 (z, y, x) -> (x, y, z)
  - PillarLayer: 整合batch数据整pillars
  - PillarEncoder: pillar 编码((N, 4) -> (N, 9) -> (1, 64))
  - PillarScatter
- Backbone
  - 完成
  - nn.init.kaiming_normal_初始化方式 (`删除cuda seed`)
- Neck
  - 完成
  - nn.init.kaiming_normal_初始化方式 (`删除cuda seed`)
- Head
  - 完成
  - nn.init.normal_(m.weight, mean=0, std=0.01)初始化方式 (`删除cuda seed`)

## Bbox (bottom center for z)

- Anchor
  - 生成完成
- 编码, 解码完成
- iou3d (完成)
  - bev overlap
  - height overlap
  - iou3d
- anchor生成label

## Train

- loss
  - cls loss
    - pos: 与0, 1, 2类bboxes具有很大iou(大于某个阈值)的anchors; 与0, 1, 2类具有最大iou的anchors.
    - neg: 与任何bboxes最有非常小iou(小于某个阈值)的anchors; 与-1类bboxes具有很大iou或者最大iou的anchors.
  - reg loss
    - 训练样本: 与0, 1, 2类bboxes具有很大iou(大于某个阈值)的anchors; 与0, 1, 2类具有最大iou的anchors.
  - dir cls loss
    - 训练样本: 与0, 1, 2类bboxes具有很大iou(大于某个阈值)的anchors; 与0, 1, 2类具有最大iou的anchors.
- 优化器
- 开始训练

## Test and evaluate

- 预测bbox
- 可视化
- 评估
  - label_2的格式, mmdet3d .pkl的格式
  - iou 计算(bbox, bev, 3dbbox)
  - gt_bbox, dt_bbox分类 (ignore, normal, remove)
  - 基于tp, 计算score threshold
  - 计算precision, recall
  - 计算AP, mAP
- 性能优化
- 代码优化 (不急)

## bbox property evaluation

以 `label='Pedestrian', difficulty=1`为例, 

| ignore level | gt | det |
|:---:|:---:|:---:|
| 1 | (cur_label == 'Pedestrian' && cur_difficulty > 1) or cur_label == 'Person_sitting' | cur_heigh < MIN_HEIGHTS[1] |
| 0 | cur_label == 'Pedestrian' && cur_difficulty <= 1 | cur_label == 'Pedestrian' |
| - 1 | Others | Others | 
