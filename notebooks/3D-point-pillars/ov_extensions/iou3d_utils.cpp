// Shared IoU 3D utility functions
#include "iou3d_utils.hpp"
// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu
// https://github.com/zhulf0804/PointPillars/blob/main/pointpillars/ops/iou3d/iou3d_kernel.cu

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr float EPS = 1e-8f;

struct Point {
  float x, y;
  Point() : x(0), y(0) {}
  Point(float _x, float _y) : x(_x), y(_y) {}
  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }
  Point operator+(const Point &b) const { return Point(x + b.x, y + b.y); }
  Point operator-(const Point &b) const { return Point(x - b.x, y - b.y); }
};

inline float cross(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1,
                            const Point &q2) {
  int ret = std::min(p1.x, p2.x) <= std::max(q1.x, q2.x) &&
            std::min(q1.x, q2.x) <= std::max(p1.x, p2.x) &&
            std::min(p1.y, p2.y) <= std::max(q1.y, q2.y) &&
            std::min(q1.y, q2.y) <= std::max(p1.y, p2.y);
  return ret;
}

inline int check_in_box2d(const float *box, const Point &p) {
  const float MARGIN = 1e-5f;
  float center_x = (box[0] + box[2]) / 2.f;
  float center_y = (box[1] + box[3]) / 2.f;
  float angle_cos = std::cos(-box[4]), angle_sin = std::sin(-box[4]);
  float rot_x =
      (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
  float rot_y =
      -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
  return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
          rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1,
                        const Point &q0, Point &ans) {
  if (check_rect_cross(p0, p1, q0, q1) == 0)
    return 0;
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);
  if (!(s1 * s2 > 0 && s3 * s4 > 0))
    return 0;

  float s5 = cross(q1, p1, p0);
  if (std::fabs(s5 - s1) > EPS) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;
    // If D == 0 lines are parallel; avoid division by zero:      // Extra
    if (std::fabs(D) < EPS) { // Extra
      // fallback: use midpoint of overlapping segment            // Extra
      ans.x = (p0.x + p1.x + q0.x + q1.x) * 0.25f; // Extra
      ans.y = (p0.y + p1.y + q0.y + q1.y) * 0.25f; // Extra
    } else {                                       // Extra
      ans.x = (b0 * c1 - b1 * c0) / D;
      ans.y = (a1 * c0 - a0 * c1) / D;
    } // Extra
  }
  return 1;
}

inline void rotate_around_center(const Point &center, const float angle_cos,
                                 const float angle_sin, Point &p) {
  float new_x =
      (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
  float new_y =
      -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

inline bool point_cmp(const Point &a, const Point &b, const Point &center) {
  return std::atan2(a.y - center.y, a.x - center.x) >
         std::atan2(b.y - center.y, b.x - center.x);
}

float box_overlap(const float *box_a, const float *box_b) {
  float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3],
        a_angle = box_a[4];
  float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3],
        b_angle = box_b[4];

  Point center_a((a_x1 + a_x2) / 2.f, (a_y1 + a_y2) / 2.f);
  Point center_b((b_x1 + b_x2) / 2.f, (b_y1 + b_y2) / 2.f);

  Point box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  float a_angle_cos = std::cos(a_angle), a_angle_sin = std::sin(a_angle);
  float b_angle_cos = std::cos(b_angle), b_angle_sin = std::sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
  }
  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;
  poly_center.set(0, 0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  float area = 0.0f;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }
  return std::fabs(area) / 2.0f;
}

float iou_bev(const float *box_a, const float *box_b) {
  float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
  float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / std::fmax(sa + sb - s_overlap, EPS);
}
