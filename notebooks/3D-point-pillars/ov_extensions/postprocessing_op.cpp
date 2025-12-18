#include "postprocessing_op.hpp"
#include "iou3d_utils.hpp"
#include <openvino/op/op.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

namespace ov {
namespace custom_ops {

PostProcessingOp::PostProcessingOp(const ov::Output<ov::Node>& bbox_cls_pred,
                                   const ov::Output<ov::Node>& bbox_pred,
                                   const ov::Output<ov::Node>& bbox_dir_cls_pred,
                                   const std::vector<float>& ranges,
                                   const std::vector<float>& sizes,
                                   const std::vector<float>& rotations,
                                   int nclasses,
                                   int nms_pre,
                                   float score_thr,
                                   float nms_thr,
                                   int max_num)
    : ov::op::Op({bbox_cls_pred, bbox_pred, bbox_dir_cls_pred}),
      m_ranges(ranges),
      m_sizes(sizes),
      m_rotations(rotations),
      m_nclasses(nclasses),
      m_nms_pre(nms_pre),
      m_score_thr(score_thr),
      m_nms_thr(nms_thr),
      m_max_num(max_num) {
    constructor_validate_and_infer_types();
}

void PostProcessingOp::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3, "PostProcessingOp expects 3 inputs");

    const auto& cls_shape = get_input_partial_shape(0);
    const auto& bbox_shape = get_input_partial_shape(1);
    const auto& dir_shape = get_input_partial_shape(2);

    OPENVINO_ASSERT(cls_shape.rank().is_static() && cls_shape.rank().get_length() == 3,
                    "bbox_cls_pred must be 3D [n_anchors*nclasses, H, W]");
    OPENVINO_ASSERT(bbox_shape.rank().is_static() && bbox_shape.rank().get_length() == 3,
                    "bbox_pred must be 3D [n_anchors*7, H, W]");
    OPENVINO_ASSERT(dir_shape.rank().is_static() && dir_shape.rank().get_length() == 3,
                    "bbox_dir_cls_pred must be 3D [n_anchors*2, H, W]");

    // Outputs: bboxes [k, 7], labels [k], scores [k]
    set_output_type(0, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 7});
    set_output_type(1, ov::element::i64, ov::PartialShape{ov::Dimension::dynamic()});
    set_output_type(2, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic()});
}

std::shared_ptr<ov::Node> PostProcessingOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");

    return std::make_shared<PostProcessingOp>(new_args[0], new_args[1], new_args[2],
                                              m_ranges, m_sizes, m_rotations,
                                              m_nclasses, m_nms_pre, m_score_thr, m_nms_thr, m_max_num);
}

bool PostProcessingOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("ranges", m_ranges);
    visitor.on_attribute("sizes", m_sizes);
    visitor.on_attribute("rotations", m_rotations);
    visitor.on_attribute("nclasses", m_nclasses);
    visitor.on_attribute("nms_pre", m_nms_pre);
    visitor.on_attribute("score_thr", m_score_thr);
    visitor.on_attribute("nms_thr", m_nms_thr);
    visitor.on_attribute("max_num", m_max_num);
    return true;
}

bool PostProcessingOp::has_evaluate() const {
    return true;
}

// Helper: limit_period for angle normalization
float PostProcessingOp::limit_period(float val, float offset, float period) const {
    return val - std::floor(val / period + offset) * period;
}

// Helper: axis-aligned IoU (used in NMS - does NOT consider rotation!)
inline float iou_normal(const float* a, const float* b) {
    float left = std::max(a[0], b[0]);
    float right = std::min(a[2], b[2]);
    float top = std::max(a[1], b[1]);
    float bottom = std::min(a[3], b[3]);
    float width = std::max(right - left, 0.0f);
    float height = std::max(bottom - top, 0.0f);
    float interS = width * height;
    float Sa = (a[2] - a[0]) * (a[3] - a[1]);
    float Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / std::max(Sa + Sb - interS, 1e-10f);
}

// Helper: decode bbox from anchor + delta
void PostProcessingOp::decode_bbox(const float* anchor, const float* delta, float* bbox) const {
    // anchor: [x, y, z, w, l, h, theta]
    // delta: [dx, dy, dz, dw, dl, dh, dtheta]
    float da = std::sqrt(anchor[3] * anchor[3] + anchor[4] * anchor[4]);

    bbox[0] = delta[0] * da + anchor[0];  // x
    bbox[1] = delta[1] * da + anchor[1];  // y
    bbox[2] = delta[2] * anchor[5] + anchor[2] + anchor[5] / 2.0f;  // z
    bbox[3] = anchor[3] * std::exp(delta[3]);  // w
    bbox[4] = anchor[4] * std::exp(delta[4]);  // l
    bbox[5] = anchor[5] * std::exp(delta[5]);  // h
    bbox[2] = bbox[2] - bbox[5] / 2.0f;  // z adjustment
    bbox[6] = anchor[6] + delta[6];  // theta
}

// Helper: generate anchors for given feature map size
void PostProcessingOp::generate_anchors(int height, int width, std::vector<float>& anchors) const {
    // anchors output: [height * width * n_classes * n_rotations, 7]
    // For PointPillars: 3 classes, 2 rotations → 6 anchors per location
    // Order: [H, W, 3 classes, 2 rotations, 7]
    // Flattened as: anchors[(h*W*6 + w*6 + cls*2 + rot) * 7]

    const int n_anchor_classes = m_ranges.size() / 6;  // Each class has 6 range values
    const int n_rotations = m_rotations.size();
    const int total_anchors = height * width * n_anchor_classes * n_rotations;

    anchors.resize(total_anchors * 7);

    // Loop order: y → x → class → rotation (to match PyTorch [H, W, 3, 2, 7])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int cls = 0; cls < n_anchor_classes; ++cls) {
                const float* range = &m_ranges[cls * 6];  // [x1, y1, z1, x2, y2, z2]
                const float* size = &m_sizes[cls * 3];    // [w, l, h]

                float x_step = (range[3] - range[0]) / width;
                float y_step = (range[4] - range[1]) / height;
                float x_shift = x_step / 2.0f;
                float y_shift = y_step / 2.0f;
                float z_center = (range[2] + range[5]) / 2.0f;

                float x_center = range[0] + x * x_step + x_shift;
                float y_center = range[1] + y * y_step + y_shift;

                for (int rot = 0; rot < n_rotations; ++rot) {
                    // Index: (h * W * 6 + w * 6 + cls * 2 + rot) * 7
                    int anchor_idx = ((y * width + x) * n_anchor_classes + cls) * n_rotations + rot;
                    float* anchor = &anchors[anchor_idx * 7];
                    anchor[0] = x_center;
                    anchor[1] = y_center;
                    anchor[2] = z_center;
                    anchor[3] = size[0];  // w
                    anchor[4] = size[1];  // l
                    anchor[5] = size[2];  // h
                    anchor[6] = m_rotations[rot];  // theta
                }
            }
        }
    }
}

bool PostProcessingOp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 3, "PostProcessingOp expects 3 inputs");
    OPENVINO_ASSERT(outputs.size() == 3, "PostProcessingOp expects 3 outputs");

    const auto& cls_tensor = inputs[0];
    const auto& bbox_tensor = inputs[1];
    const auto& dir_tensor = inputs[2];

    const auto cls_shape = cls_tensor.get_shape();  // [n_anchors*nclasses, H, W]
    const int height = cls_shape[1];
    const int width = cls_shape[2];
    const int n_anchor_per_loc = 6;  // 3 classes * 2 rotations
    const int total_anchors = height * width * n_anchor_per_loc;

    const float* cls_data = cls_tensor.data<float>();
    const float* bbox_data = bbox_tensor.data<float>();
    const float* dir_data = dir_tensor.data<float>();

    // Step 1: Generate anchors
    std::vector<float> anchors;
    // std::cout << "Generating anchors for feature map size: " << height << "x" << width << std::endl;
    generate_anchors(height, width, anchors);

    // Step 2: Transpose and reshape predictions
    // cls_pred: [n_anchors*nclasses, H, W] → [H, W, n_anchors*nclasses] → [total_anchors, nclasses]
    std::vector<float> cls_reshaped(total_anchors * m_nclasses);
    std::vector<float> bbox_reshaped(total_anchors * 7);
    std::vector<float> dir_reshaped(total_anchors * 2);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int a = 0; a < n_anchor_per_loc; ++a) {
                int out_idx = (h * width + w) * n_anchor_per_loc + a;

                // Classification scores (per class)
                for (int c = 0; c < m_nclasses; ++c) {
                    int in_idx = ((a * m_nclasses + c) * height + h) * width + w;
                    cls_reshaped[out_idx * m_nclasses + c] = cls_data[in_idx];
                }

                // BBox deltas
                for (int d = 0; d < 7; ++d) {
                    int in_idx = ((a * 7 + d) * height + h) * width + w;
                    bbox_reshaped[out_idx * 7 + d] = bbox_data[in_idx];
                }

                // Direction classification
                for (int d = 0; d < 2; ++d) {
                    int in_idx = ((a * 2 + d) * height + h) * width + w;
                    dir_reshaped[out_idx * 2 + d] = dir_data[in_idx];
                }
            }
        }
    }

    // Step 3: Apply sigmoid to classification scores
    for (size_t i = 0; i < cls_reshaped.size(); ++i) {
        cls_reshaped[i] = 1.0f / (1.0f + std::exp(-cls_reshaped[i]));
    }

    // Step 4: Get direction class (argmax)
    std::vector<int> dir_classes(total_anchors);
    for (int i = 0; i < total_anchors; ++i) {
        dir_classes[i] = dir_reshaped[i * 2] > dir_reshaped[i * 2 + 1] ? 0 : 1;
    }

    // Step 5: Select top nms_pre boxes by max class score
    std::vector<std::pair<float, int>> score_indices;
    for (int i = 0; i < total_anchors; ++i) {
        float max_score = 0.0f;
        for (int c = 0; c < m_nclasses; ++c) {
            max_score = std::max(max_score, cls_reshaped[i * m_nclasses + c]);
        }
        score_indices.push_back({max_score, i});
    }

    std::partial_sort(score_indices.begin(),
                     score_indices.begin() + std::min(m_nms_pre, (int)score_indices.size()),
                     score_indices.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    int nms_pre_actual = std::min(m_nms_pre, (int)score_indices.size());

    // Step 6: Decode bboxes for selected indices
    std::vector<float> decoded_bboxes(nms_pre_actual * 7);
    std::vector<float> selected_cls_scores(nms_pre_actual * m_nclasses);
    std::vector<int> selected_dir_classes(nms_pre_actual);

    for (int i = 0; i < nms_pre_actual; ++i) {
        int idx = score_indices[i].second;
        decode_bbox(&anchors[idx * 7], &bbox_reshaped[idx * 7], &decoded_bboxes[i * 7]);
        for (int c = 0; c < m_nclasses; ++c) {
            selected_cls_scores[i * m_nclasses + c] = cls_reshaped[idx * m_nclasses + c];
        }
        selected_dir_classes[i] = dir_classes[idx];
    }

    // Step 7: Per-class NMS using mask-based algorithm (matching PyTorch exactly)
    std::vector<float> final_bboxes;
    std::vector<int64_t> final_labels;
    std::vector<float> final_scores;

    const int THREADS_PER_BLOCK_NMS = 64;  // Similar to PyTorch

    for (int cls = 0; cls < m_nclasses; ++cls) {
        // Filter by score threshold
        std::vector<int> cls_indices;
        std::vector<float> cls_scores;

        for (int i = 0; i < nms_pre_actual; ++i) {
            float score = selected_cls_scores[i * m_nclasses + cls];
            if (score > m_score_thr) {
                cls_indices.push_back(i);
                cls_scores.push_back(score);
            }
        }

        if (cls_indices.empty()) continue;

        int boxes_num = cls_indices.size();

        // Sort by score descending (same as PyTorch)
        std::vector<int> order(boxes_num);
        for (int i = 0; i < boxes_num; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                 [&cls_scores](int a, int b) { return cls_scores[a] > cls_scores[b]; });

        // Create sorted arrays of boxes (in 2D format) to match PyTorch's nms_kernel input
        std::vector<float> sorted_boxes_2d(boxes_num * 5);
        for (int i = 0; i < boxes_num; ++i) {
            int orig_idx = cls_indices[order[i]];
            const float* bbox = &decoded_bboxes[orig_idx * 7];
            sorted_boxes_2d[i * 5 + 0] = bbox[0] - bbox[3] / 2.0f;  // xmin
            sorted_boxes_2d[i * 5 + 1] = bbox[1] - bbox[4] / 2.0f;  // ymin
            sorted_boxes_2d[i * 5 + 2] = bbox[0] + bbox[3] / 2.0f;  // xmax
            sorted_boxes_2d[i * 5 + 3] = bbox[1] + bbox[4] / 2.0f;  // ymax
            sorted_boxes_2d[i * 5 + 4] = bbox[6];                   // theta
        }

        // Apply mask-based NMS (similar to PyTorch nms_kernel)
        const int col_blocks = (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
        std::vector<unsigned long long> mask(boxes_num * col_blocks, 0);

        // Compute overlap mask
        for (int row_start = 0; row_start < col_blocks; ++row_start) {
            for (int col_start = 0; col_start < col_blocks; ++col_start) {
                int row_size = std::min(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
                int col_size = std::min(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

                for (int t = 0; t < row_size; ++t) {
                    int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + t;
                    const float* cur_box = &sorted_boxes_2d[cur_box_idx * 5];
                    unsigned long long u = 0ULL;
                    int start = (row_start == col_start) ? (t + 1) : 0;

                    for (int i = start; i < col_size; ++i) {
                        int other_idx = THREADS_PER_BLOCK_NMS * col_start + i;
                        const float* other = &sorted_boxes_2d[other_idx * 5];
                        if (iou_bev(cur_box, other) > m_nms_thr) {
                            u |= (1ULL << i);
                        }
                    }
                    mask[cur_box_idx * col_blocks + col_start] = u;
                }
            }
        }

        // Extract keep indices from mask (same as PyTorch)
        std::vector<unsigned long long> remv(col_blocks, 0);
        std::vector<int> keep_indices;

        for (int i = 0; i < boxes_num; ++i) {
            int nblock = i / THREADS_PER_BLOCK_NMS;
            int inblock = i % THREADS_PER_BLOCK_NMS;

            bool is_removed = (remv[nblock] & (1ULL << inblock)) != 0;
            // if (cls == 2 && i < 10) {
            //     std::printf("[DEBUG] Box %d: nblock=%d inblock=%d remv[%d]=0x%016llx is_removed=%d\n",
            //                i, nblock, inblock, nblock, remv[nblock], is_removed);
            // }

            if (!is_removed) {
                keep_indices.push_back(i);  // This is already sorted index
                unsigned long long* p = &mask[i * col_blocks];
                for (int j = nblock; j < col_blocks; ++j) {
                    remv[j] |= p[j];
                }
            }
        }

        // Add kept boxes to final results
        for (int sorted_idx : keep_indices) {
            int orig_idx = cls_indices[order[sorted_idx]];
            const float* bbox_i = &decoded_bboxes[orig_idx * 7];

            // Apply direction adjustment
            float theta = limit_period(bbox_i[6], 1.0f, M_PI);
            theta += (1 - selected_dir_classes[orig_idx]) * M_PI;

            // Add to results
            final_bboxes.insert(final_bboxes.end(), bbox_i, bbox_i + 7);
            final_bboxes.back() = theta;  // Update with adjusted theta
            final_labels.push_back(cls);
            final_scores.push_back(cls_scores[order[sorted_idx]]);
        }
    }

    // Step 8: Apply max_num filtering (limit total detections)
    if (final_labels.size() > (size_t)m_max_num) {
        // Sort by score descending and keep top max_num
        std::vector<std::pair<float, int>> score_idx;
        for (size_t i = 0; i < final_labels.size(); ++i) {
            score_idx.push_back({final_scores[i], i});
        }
        std::partial_sort(score_idx.begin(),
                         score_idx.begin() + m_max_num,
                         score_idx.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        // Keep only top max_num
        std::vector<float> filtered_bboxes;
        std::vector<int64_t> filtered_labels;
        std::vector<float> filtered_scores;

        for (int i = 0; i < m_max_num; ++i) {
            int idx = score_idx[i].second;
            filtered_bboxes.insert(filtered_bboxes.end(),
                                  &final_bboxes[idx * 7],
                                  &final_bboxes[idx * 7 + 7]);
            filtered_labels.push_back(final_labels[idx]);
            filtered_scores.push_back(final_scores[idx]);
        }

        final_bboxes = filtered_bboxes;
        final_labels = filtered_labels;
        final_scores = filtered_scores;
    }

    // Step 9: Set outputs
    int num_detections = final_labels.size();

    outputs[0].set_shape({(size_t)num_detections, 7});
    outputs[1].set_shape({(size_t)num_detections});
    outputs[2].set_shape({(size_t)num_detections});

    if (num_detections > 0) {
        std::memcpy(outputs[0].data<float>(), final_bboxes.data(), final_bboxes.size() * sizeof(float));
        std::memcpy(outputs[1].data<int64_t>(), final_labels.data(), final_labels.size() * sizeof(int64_t));
        std::memcpy(outputs[2].data<float>(), final_scores.data(), final_scores.size() * sizeof(float));
    }

    return true;
}

}  // namespace custom_ops
}  // namespace ov
