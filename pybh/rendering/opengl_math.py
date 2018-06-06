import numpy as np


def look_at(eye, center, up, dtype=np.float32):
    eye = np.asarray(eye, dtype=dtype)
    center = np.asarray(center, dtype=dtype)
    up = np.asarray(up, dtype=dtype)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    # Compute side vector of camera axis
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    # Recompute up vector
    up = np.cross(side, forward)
    up /= np.linalg.norm(up)
    # Update view matrix
    view_mat = np.eye(4, dtype=dtype)
    view_mat[:3, 0] = side
    view_mat[:3, 1] = up
    view_mat[:3, 2] = -forward
    view_mat[:3, 3] = -eye
    return view_mat


def perspective_matrix_bbox(left, right, bottom, top, near, far, dtype=np.float32):
    return np.array([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, - (far + near) / (far - near), - 2 * near * far / (far - near)],
        [0, 0, -1, 0]
    ], dtype=dtype)


def fov_from_focal_length(f):
    fov = 2 * np.atan(1 / f)
    return fov


def focal_length_from_fov(fov):
    f = 1 / np.tan(fov / 2)
    return f


def perspective_matrix_focal_length(f, aspect, near, far, dtype=np.float32):
    """
        f: focal length
        aspect: width / height
        near: near plane
        far: far plane
    """
    return np.array([
        [f, 0, 0, 0],
        [0, f * aspect, 0, 0],
        [0, 0, - (far + near) / (far - near), - 2 * near * far / (far - near)],
        [0, 0, -1, 0]
    ], dtype=dtype)


def perspective_matrix(fov, aspect, near, far, dtype=np.float32):
    f = focal_length_from_fov(fov)
    return perspective_matrix_focal_length(f, aspect, near, far, dtype=dtype)
