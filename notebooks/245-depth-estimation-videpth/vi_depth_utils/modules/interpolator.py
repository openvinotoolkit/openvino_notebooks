import numpy as np
np.set_printoptions(suppress=True)

from scipy.interpolate import griddata


def interpolate_knots(map_size, knot_coords, knot_values, interpolate, fill_corners):
    grid_x, grid_y = np.mgrid[0:map_size[0], 0:map_size[1]]

    interpolated_map = griddata(
        points=knot_coords.T,
        values=knot_values,
        xi=(grid_y, grid_x),
        method=interpolate,
        fill_value=1.0)

    return interpolated_map


class Interpolator2D(object):
    def __init__(self, pred_inv, sparse_depth_inv, valid):
        self.pred_inv = pred_inv
        self.sparse_depth_inv = sparse_depth_inv
        self.valid = valid

        self.map_size = np.shape(pred_inv)
        self.num_knots = np.sum(valid)
        nonzero_y_loc = np.nonzero(valid)[0]
        nonzero_x_loc = np.nonzero(valid)[1]
        self.knot_coords = np.stack((nonzero_x_loc, nonzero_y_loc))
        self.knot_scales = sparse_depth_inv[valid] / pred_inv[valid]
        self.knot_shifts = sparse_depth_inv[valid] - pred_inv[valid]

        self.knot_list = []
        for i in range(self.num_knots):
            self.knot_list.append((int(self.knot_coords[0,i]), int(self.knot_coords[1,i])))

        # to be computed
        self.interpolated_map = None
        self.confidence_map = None
        self.output = None

    def generate_interpolated_scale_map(self, interpolate_method, fill_corners=False):
        self.interpolated_scale_map = interpolate_knots(
            map_size=self.map_size, 
            knot_coords=self.knot_coords, 
            knot_values=self.knot_scales,
            interpolate=interpolate_method,
            fill_corners=fill_corners
        ).astype(np.float32)