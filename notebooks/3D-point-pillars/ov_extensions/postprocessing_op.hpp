#pragma once

#include <openvino/op/op.hpp>

namespace ov {
namespace custom_ops {

/**
 * @brief PostProcessingOp: End-to-end post-processing for PointPillars
 * 
 * Inputs:
 *   - bbox_cls_pred: [n_anchors*3, H, W] - classification logits
 *   - bbox_pred: [n_anchors*7, H, W] - bbox delta predictions
 *   - bbox_dir_cls_pred: [n_anchors*2, H, W] - direction classification logits
 * 
 * Outputs:
 *   - bboxes: [k, 7] - predicted bboxes (x, y, z, w, l, h, theta)
 *   - labels: [k] - class labels (int64)
 *   - scores: [k] - confidence scores
 * 
 * Attributes:
 *   - ranges: [[x1,y1,z1,x2,y2,z2], ...] for each class (flattened array)
 *   - sizes: [[w,l,h], ...] for each class (flattened array)
 *   - rotations: [0, 1.57] rotation angles
 *   - nclasses: number of classes (3 for Car, Pedestrian, Cyclist)
 *   - nms_pre: number of top boxes to keep before NMS
 *   - score_thr: score threshold for filtering
 *   - nms_thr: IoU threshold for NMS
 */
class PostProcessingOp : public ov::op::Op {
public:
    OPENVINO_OP("PostProcessingOp", "custom_ops");

    PostProcessingOp() = default;

    PostProcessingOp(const ov::Output<ov::Node>& bbox_cls_pred,
                     const ov::Output<ov::Node>& bbox_pred,
                     const ov::Output<ov::Node>& bbox_dir_cls_pred,
                     const std::vector<float>& ranges,
                     const std::vector<float>& sizes,
                     const std::vector<float>& rotations,
                     int nclasses,
                     int nms_pre,
                     float score_thr,
                     float nms_thr,
                     int max_num);

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool has_evaluate() const override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;

private:
    // Anchor configuration
    std::vector<float> m_ranges;     // Flattened: 3 classes * 6 values
    std::vector<float> m_sizes;      // Flattened: 3 classes * 3 values
    std::vector<float> m_rotations;  // [0, 1.57]
    
    // Post-processing parameters
    int m_nclasses;
    int m_nms_pre;
    float m_score_thr;
    float m_nms_thr;
    int m_max_num;

    // Helper functions
    void generate_anchors(int height, int width, std::vector<float>& anchors) const;
    void decode_bbox(const float* anchor, const float* delta, float* bbox) const;
    float limit_period(float val, float offset, float period) const;
};

}  // namespace custom_ops
}  // namespace ov
