import numpy as np
from .contrib import transformations


def degrees_to_radians(degrees):
    """Convert angle in degrees to radians"""
    return degrees * np.pi / 180.0


def radians_to_degrees(radians):
    """Convert angle in radians to degrees"""
    return radians * 180.0 / np.pi


class BoundingBox(object):

    def __init__(self, min, max):
        self._min = np.array(min)
        self._max = np.array(max)

    @staticmethod
    def empty(dtype=np.float):
        return BoundingBox(
            np.array([1, 1, 1], dtype=dtype),
            np.array([-1, -1, -1], dtype=dtype),
        )

    @staticmethod
    def infinite(dtype=np.float):
        return BoundingBox(
            np.array([-np.inf, -np.inf, -np.inf], dtype=dtype),
            np.array([np.inf, np.inf, np.inf], dtype=dtype))

    @staticmethod
    def max_extent(dtype=np.float):
        min_value = - np.finfo(dtype).max
        max_value = np.finfo(dtype).max
        return BoundingBox(
            np.array([min_value, min_value, min_value], dtype=dtype),
            np.array([max_value, max_value, max_value], dtype=dtype))

    @staticmethod
    def from_center_and_extent(center, extent):
        half_extent = 0.5 * extent
        return BoundingBox(center - half_extent, center + half_extent)

    def minimum(self):
        return self._min

    def maximum(self):
        return self._max

    def extent(self):
        return self._max - self._min

    def min_extent(self):
        return np.min(self._max - self._min)

    def max_extent(self):
        return np.max(self._max - self._min)

    def contains(self, xyz):
        if isinstance(xyz, BoundingBox):
            return np.all(xyz.minimum() >= self.minimum()) \
                   and np.all(xyz.maximum() <= self.maximum())
        else:
            return np.all(xyz >= self._min) and np.all(xyz <= self._max)

    def move(self, offset):
        return BoundingBox(self._min + offset, self._max + offset)

    def move_in_place(self, offset):
        self._min += offset
        self._max += offset
        return self

    def scale(self, factor):
        center = 0.5 * (self._min + self._max)
        extent = self._max - self._min
        return BoundingBox.from_center_and_extent(center, factor * extent)

    def scale_in_place(self, factor):
        center = 0.5 * (self._min + self._max)
        extent = self._max - self._min
        self._min = center - factor * extent
        self._max = center + factor * extent
        return self

    def __str__(self):
        return "({}, {})".format(self.minimum(), self.maximum())


def convert_xyz_from_left_to_right_handed(location):
    """Convert xyz position from left- to right-handed coordinate system"""
    location = np.array([location[0], -location[1], location[2]])
    return location


def convert_xyz_from_right_to_left_handed(location):
    """Convert xyz position from right- to left-handed coordinate system"""
    location = np.array([location[0], -location[1], location[2]])
    return location


def convert_rpy_from_left_to_right_handed(orientation_rpy):
    """Convert roll, pitch, yaw euler angles from left- to right-handed coordinate system"""
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


def convert_rpy_from_right_to_left_handed(orientation_rpy):
    """Convert roll, pitch, yaw euler angles from right- to left-handed coordinate system"""
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


def convert_rpy_to_quat(orientation_rpy):
    """Convert roll, pitch, yaw euler angles to quaternion"""
    roll, pitch, yaw = orientation_rpy
    quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
    return quat


def convert_quat_to_rpy(quat):
    """Convert quaternion to euler angles in radians"""
    yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
    return roll, pitch, yaw


def normalize_quaternion(quat):
  """Return normalized quaternion (qx, qy, qz, qw)"""
  quat_norm = np.sqrt(np.sum(quat ** 2))
  return quat / quat_norm


def invert_quaternion(quat):
  """Rotate a given quaternion (qx, qy, qz, qw)"""
  return transformations.quaternion_inverse(quat)


def multiply_quaternion(quat1, quat2):
    """Multiply two quaterions (qx, qy, qz, qw)"""
    return transformations.quaternion_multiply(quat1, quat2)


def rotate_vector_with_quaternion(quat, vec):
    """Rotate a vector with a given quaternion (x, y, z, w)"""
    vec_q = [vec[0], vec[1], vec[2], 0]
    rot_vec_q = transformations.quaternion_multiply(
        transformations.quaternion_multiply(quat, vec_q),
        transformations.quaternion_conjugate(quat))
    rot_vec = rot_vec_q[:3]
    return rot_vec


def rotate_vector_with_rpy(orientation_rpy, vec):
    """Rotate a vector with a given rpy orientation (roll, pitch, yaw)"""
    quat = convert_rpy_to_quat(orientation_rpy)
    return rotate_vector_with_quaternion(quat, vec)


def is_angle_equal(angle1, angle2, tolerance=1e-10):
    """Compare if two angles in radians are equal according to a tolerance"""
    d_angle = np.abs(angle1 - angle2)
    d_angle = d_angle % (2 * np.pi)
    assert(d_angle >= 0)
    assert(d_angle < 2 * np.pi)
    if d_angle > np.pi:
        d_angle = 2 * np.pi - d_angle
    return d_angle < tolerance


def is_angle_equal_degrees(angle1, angle2, tolerance=1e-10):
    """Compare if two angles in degrees are equal according to a tolerance"""
    return is_angle_equal(degrees_to_radians(angle1), degrees_to_radians(angle2), tolerance)


def is_vector_equal_cwise(vec1, vec2, tolerance=1e-10):
    """Compare if two vectors are equal component-wise according to a tolerance"""
    return np.all(np.abs(vec1 - vec2) <= tolerance)


def is_vector_equal(vec1, vec2, tolerance=1e-10):
    """Compare if two vectors are equal (L2-norm) according to a tolerance"""
    return np.linalg.norm(vec1 - vec2) <= tolerance


def is_quaternion_equal(quat1, quat2, tolerance=1e-10):
    """Compare if two quaternions are equal

    This depends on component-wise comparison. A better way would be to use the angular difference
    """
    return is_vector_equal_cwise(quat1, quat2, tolerance) \
        or is_vector_equal_cwise(quat1, -quat2, tolerance)


class SinglePassStatistics(object):

    def __init__(self, size=None, dtype=np.float):
        self._mean = np.zeros(size, dtype=dtype)
        self._variance_acc = np.zeros(size, dtype=dtype)
        self._min = None
        self._max = None
        self._N = 0

    def add_value(self, value):
        if self._N == 0:
            self._min = value
            self._max = value
        self._N += 1
        prev_mean = self._mean
        self._mean += (value - self._mean) / self._N
        self._variance_acc += (value - self._mean) * (value - prev_mean)
        self._min = np.minimum(value, self._min)
        self._max = np.maximum(value, self._max)

    def add_values(self, values):
        raise NotImplementedError()
        # self._N += len(values)
        # prev_mean = self._mean
        # self._mean += (np.sum(values) - len(values) * self._mean) / self._N
        # Do variance calculation
        # self._variance_acc += (value - self._mean) * (value - prev_mean)
        # self._min = min(self._min, np.min(value))
        # self._max = max(self._max, np.max(value))

    @property
    def mean(self):
        return np.asarray(self._mean)

    @property
    def variance(self):
        return np.asarray(self._variance_acc / float(self._N - 1))

    @property
    def stddev(self):
        return np.asarray(np.sqrt(self.variance))

    @property
    def min(self):
        return np.asarray(self._min)

    @property
    def max(self):
        return np.asarray(self._max)

    @property
    def num_samples(self):
        return self._N
