import os
import numpy as np
import open3d as o3d

def read_kitti_pcd(pcd_path):
    """
    Read a binary PCD written as FIELDS x y z intensity and return Nx4 float32 array.

    Strategy:
      1. Try Open3D tensor API (o3d.t.io.read_point_cloud) — preserves attributes including 'intensity'.
      2. If intensity is not available from Open3D, fall back to the robust parser
         (pointpillars.dataset.kitti_open3d.utils.read_pcd_xyzi) which reads the binary payload directly.

    Returns:
      Nx4 numpy float32 array: columns x, y, z, intensity (intensity filled with 0.0 if absent).
    """

    # 1) Try Open3D tensor API (o3d.t.io.read_point_cloud)
    try:
        pcd_t = o3d.t.io.read_point_cloud(pcd_path)  # returns o3d.t.geometry.PointCloud-like

        # positions
        pos = pcd_t.point.positions.numpy().astype(np.float32)

        # intensity
        if "intensity" in pcd_t.point:
            inten = pcd_t.point["intensity"].numpy().astype(np.float32).reshape(-1, 1)
        else:
            # no intensity attribute available in tensor API -> pad zeros
            inten = np.zeros((pos.shape[0], 1), dtype=np.float32)

        return np.concatenate([pos, inten], axis=1)
    except Exception:
        # Something unexpected in tensor read; fall through to legacy
        print(f"Warning: Open3D tensor read failed for {pcd_path}, falling back to custom reader.")

    # 2) Guaranteed fallback: parse the PCD file directly to guarantee [x,y,z,intensity]
    return read_pcd_xyzi(pcd_path)


def read_pcd_xyzi(pcd_path):
    """Read a binary PCD file with FIELDS x y z intensity and return Nx4 float32 array.

    This is the inverse of convert_bin_to_pcd() and returns exactly the same
    data structure as reading a KITTI .bin file with np.fromfile().reshape(-1,4).
    """
    with open(pcd_path, 'rb') as f:
        # Read header lines until DATA
        n_points = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('POINTS'):
                n_points = int(line.split()[1])
            if line.startswith('DATA'):
                break

        if n_points is None:
            raise ValueError(f"Could not find POINTS in PCD header: {pcd_path}")

        # Read binary payload
        binary_data = f.read()

    pts = np.frombuffer(binary_data, dtype=np.float32).reshape(n_points, 4).copy()
    return pts


def convert_bin_to_pcd(bin_path, pcd_path):
    """Convert KITTI .bin (x,y,z,intensity float32) to binary PCD preserving ALL fields.

    Writes a proper PCD with FIELDS x y z intensity so intensity is preserved
    as a numeric scalar field (not just colors). Compatible with PCL and other
    PCD readers that support scalar fields.
    """
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    N = pts.shape[0]

    # Write binary PCD with header describing all 4 fields
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    )

    with open(pcd_path, 'wb') as f:
        f.write(header.encode('ascii'))
        # Write raw binary float32 data (little-endian, C-contiguous)
        f.write(pts.astype(np.float32).tobytes(order='C'))


def compare_bin_pcd_pts(pts_bin, pts_pcd, rtol=1e-5, atol=1e-8, verbose=True):
    """Compare .bin and .pcd files for data equality.

    Returns True if shapes match and all values are equal within tolerance.
    Prints summary if verbose=True.
    """
    if pts_bin.shape != pts_pcd.shape:
        if verbose:
            print(f"✗ Shape mismatch: bin={pts_bin.shape}, pcd={pts_pcd.shape}")
        return False

    if np.allclose(pts_bin, pts_pcd, rtol=rtol, atol=atol):
        if verbose:
            print(f"✓ Data matches: {pts_bin.shape[0]} points, all values within tolerance")
            print(f"  Intensity range: [{pts_bin[:,3].min():.4f}, {pts_bin[:,3].max():.4f}]")
        return True

    # Report differences
    diff = np.abs(pts_bin - pts_pcd)
    n_diff = np.sum(~np.isclose(pts_bin, pts_pcd, rtol=rtol, atol=atol))
    if verbose:
        print(f"✗ Data mismatch: {n_diff} values differ, max diff={diff.max():.6g}")
    return False


def compare_bin_pcd_file(bin_path, pcd_path, rtol=1e-5, atol=1e-8, verbose=True):
    """Compare .bin and .pcd files for data equality.

    Returns True if shapes match and all values are equal within tolerance.
    Prints summary if verbose=True.
    """
    # Read .bin
    pts_bin = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # Read .pcd
    pts_pcd = read_kitti_pcd(pcd_path)

    return compare_bin_pcd_pts(pts_bin, pts_pcd, rtol, atol, verbose)


if __name__ == "__main__":
    # Simple test: convert a .bin to .pcd and back, then compare
    import tempfile
    import sys

    # Default demo .bin in the repo (used when no path is provided)
    default_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo_data', 'test', '000002.bin'))

    if len(sys.argv) == 1:
        bin_path = default_bin
        print(f"No input provided, using default demo file: {bin_path}")
    elif len(sys.argv) == 2:
        bin_path = sys.argv[1]
    else:
        print(f"Usage: {sys.argv[0]} [path_to_bin_file]", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(bin_path):
        print(f"File not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        pcd_path = os.path.join(tmpdir, "temp.pcd")
        convert_bin_to_pcd(bin_path, pcd_path)
        print(f"Converted {bin_path} -> {pcd_path}")

        match = compare_bin_pcd_file(bin_path, pcd_path, verbose=True)
        if match:
            print("✓ Conversion verified: .bin and .pcd data match")
        else:
            print("✗ Conversion verification failed: data mismatch", file=sys.stderr)
            sys.exit(1)
