import numpy as np
from .contrib import transformations
from . import math_utils


def translation_matrix(offset):
    mat = np.eye(4, dtype=offset.dtype)
    mat[:3, 3] = offset
    return mat


def apply_translation(mat, offset):
    offset_mat = translation_matrix(offset)
    print(offset, offset_mat)
    mat[...] = np.dot(mat, offset_mat)
    return mat


def scale_matrix(scale):
    scale = np.asarray(scale)
    if len(scale) <= 1:
        scale = np.asarray([scale, scale, scale])
    mat = np.eye(4, dtype=scale.dtype)
    print(scale, mat)
    mat[:3, :3] = np.diag(scale)
    return mat


def apply_scale(mat, scale):
    scale_mat = scale_matrix(scale)
    mat[...] = np.dot(mat, scale_mat)
    return mat


def x_rotation_matrix(angle, point=None):
    rot_mat = transformations.rotation_matrix(angle, [1, 0, 0], point)
    return rot_mat


def apply_x_rotation(mat, angle, point=None):
    rot_mat = x_rotation_matrix(angle, point)
    mat[...] = np.dot(mat, rot_mat)
    return mat


def y_rotation_matrix(angle, point=None):
    rot_mat = transformations.rotation_matrix(angle, [0, 1, 0], point)
    return rot_mat


def apply_y_rotation(mat, angle, point=None):
    rot_mat = y_rotation_matrix(angle, point)
    mat[...] = np.dot(mat, rot_mat)
    return mat


def z_rotation_matrix(angle, point=None):
    rot_mat = transformations.rotation_matrix(angle, [0, 0, 1], point)
    return rot_mat


def apply_z_rotation(mat, angle, point=None):
    rot_mat = z_rotation_matrix(angle, point)
    mat[...] = np.dot(mat, rot_mat)
    return mat


def rotation_matrix(angle, axis, point=None):
    rot_mat = transformations.rotation_matrix(angle, axis, point)
    return rot_mat


def apply_rotation(mat, angle, axis, point=None):
    rot_mat = rotation_matrix(angle, axis, point)
    mat[...] = np.dot(mat, rot_mat)
    return mat


class RotationMatrix(object):
    """A rotation represented by a matrix"""

    @staticmethod
    def identity():
        return RotationMatrix(np.eye(3))

    @staticmethod
    def axis_rotation(angle, axis, point=None):
        return RotationMatrix(transformations.rotation_matrix(angle, axis, point)[:3, :3])

    @staticmethod
    def x_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[1, 0, 0])

    @staticmethod
    def y_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[0, 1, 0])

    @staticmethod
    def z_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[0, 0, 1])

    def __init__(self, mat, copy=False):
        if copy:
            mat = np.copy(mat)
        assert mat.shape == (3, 3)
        self._mat = mat

    def apply_to(self, rotation):
        return RotationMatrix(np.dot(self._mat, rotation.get_matrix(copy=False)))

    def apply_to_vector(self, vec):
        assert len(vec) == 3
        return np.dot(self._mat, vec)

    def get_inverse(self):
        return RotationMatrix(np.linalg.inv(self._mat))

    def invert(self):
        self._mat = np.linalg.inv(self._mat)

    def get_transpose(self):
        return RotationMatrix(np.transpose(self._mat))

    def transpose(self):
        self._mat = np.transpose(self._mat)

    def get_matrix(self, copy=True):
        if copy:
            return np.copy(self._mat)
        else:
            return self._mat

    @property
    def matrix(self):
        return self.get_matrix()

    def get_quaternion(self, copy=True):
        return transformations.quaternion_from_matrix(self._mat)

    @property
    def quaternion(self):
        return self.get_quaternion()


class Rotation(object):
    """A rotation represented by a quaternion (qx, qy, qz, qw)"""

    @staticmethod
    def identity():
        return Rotation(np.array([0, 0, 0, 1]))

    @staticmethod
    def axis_rotation(angle, axis, point=None):
        return Rotation(RotationMatrix.axis_rotation(angle, axis, point).quaternion)

    @staticmethod
    def x_rotation(angle):
        return Rotation(RotationMatrix.x_rotation(angle).quaternion)

    @staticmethod
    def y_rotation(angle):
        return Rotation(RotationMatrix.y_rotation(angle).quaternion)

    @staticmethod
    def z_rotation(angle):
        return Rotation(RotationMatrix.z_rotation(angle).quaternion)

    def __init__(self, quaternion, copy=False):
        if copy:
            quaternion = np.copy(quaternion)
        self._quaternion = quaternion

    def apply_to(self, rotation):
        return Rotation(math_utils.multiply_quaternion(self._quaternion, rotation.quaternion))

    def apply_to_quat(self, quat):
        assert len(quat) == 4
        return math_utils.multiply_quaternion(self._quaternion, quat)

    def apply_to_vector(self, vec):
        assert len(vec) == 3
        return math_utils.rotate_vector_with_quaternion(self._quaternion, vec)

    def get_inverse(self):
        return Rotation(math_utils.invert_quaternion(self._quaternion))

    def invert(self):
        self._quaternion = math_utils.invert_quaternion(self._quaternion)

    def get_quaternion(self, copy=True):
        if copy:
            return np.copy(self._quaternion)
        else:
            return self._quaternion

    @property
    def quaternion(self):
        return self.get_quaternion()

    def get_matrix(self, copy=True):
        return transformations.quaternion_matrix(self._quaternion)[:3, :3]

    @property
    def matrix(self):
        return self.get_matrix()


class Transform(object):
    """A rigid transformation represented by a translation vector and rotation quaternion"""

    @staticmethod
    def from_transformation_matrix(transformation_mat):
        transformation_mat = np.asarray(transformation_mat)
        transform = TransformMatrix(transformation_mat)
        return Transform(transform.translation, transform.quaternion)

    @staticmethod
    def from_translation_rotation_matrix(translation, rotation_mat):
        translation = np.asarray(translation)
        rotation_mat = np.asarray(rotation_mat)
        quat = RotationMatrix(rotation_mat).quaternion
        return Transform(translation, quat)

    @staticmethod
    def from_translation_quaternion(translation, quat):
        translation = np.asarray(translation)
        quat = np.asarray(quat)
        return Transform(translation, quat)

    @staticmethod
    def from_translation(translation):
        translation = np.asarray(translation)
        quat = Rotation.identity().quaternion
        return Transform(translation, quat)

    @staticmethod
    def from_rotation_matrix(rotation_mat):
        rotation_mat = np.asarray(rotation_mat)
        translation = np.zeros((3,), dtype=rotation_mat.dtype)
        quat = RotationMatrix(rotation_mat).quaternion
        return Transform(translation, quat)

    @staticmethod
    def from_quaternion(quat):
        quat = np.asarray(quat)
        translation = np.zeros((3,), dtype=quat.dtype)
        quat = Rotation.identity().quaternion
        return Transform(translation, quat)

    @staticmethod
    def from_axis_rotation(angle, axis, point=None):
        return Transform.from_transformation_matrix(TransformMatrix.from_axis_rotation(angle, axis, point).matrix)

    @staticmethod
    def from_x_rotation(angle):
        return Transform.axis_rotation(angle, axis=[1, 0, 0])

    @staticmethod
    def from_y_rotation(angle):
        return Transform.axis_rotation(angle, axis=[0, 1, 0])

    @staticmethod
    def from_z_rotation(angle):
        return Transform.axis_rotation(angle, axis=[0, 0, 1])

    @staticmethod
    def identity(dtype=np.float):
        return Transform(np.array([0, 0, 0], dtype=dtype), np.array([0, 0, 0, 1], dtype=dtype))

    def __init__(self, translation, quaternion, copy=False):
        if copy:
            translation = np.copy(translation)
            quaternion = np.copy(quaternion)
        self._translation = translation
        self._quaternion = quaternion

    def get_inverse(self):
        inverse_trans = Transform(self.translation, self.quaternion)
        inverse_trans.invert()
        return inverse_trans

    def invert(self):
        self._quaternion = math_utils.invert_quaternion(self._quaternion)
        self._translation = - math_utils.rotate_vector_with_quaternion(self._quaternion, self._translation)

    def apply_to(self, transform):
        assert isinstance(transform, Transform)
        new_quaternion = math_utils.multiply_quaternion(self._quaternion, transform.get_quaternion(copy=False))
        new_translation = self.apply_to_vector(transform.get_translation(copy=False)) + self._translation
        return Transform(new_translation, new_quaternion)

    def apply_to_vector(self, vec):
        assert len(vec) == 3
        return math_utils.rotate_vector_with_quaternion(self._quaternion, vec)

    def get_translation(self, copy=True):
        if copy:
            return np.copy(self._transformation_mat[3, :3])
        else:
            return self._transformation_mat[3, :3]

    @property
    def translation(self):
        return self.get_translation()

    def get_rotation_matrix(self, copy=True):
        return Rotation(self._quaternion).matrix

    @property
    def rotation_matrix(self):
        return self.get_rotation_matrix()

    def get_quaternion(self, copy=True):
        if copy:
            return np.copy(self._quaternion)
        else:
            return self._quaternion

    @property
    def quaternion(self):
        return self.get_quaternion()

    def get_matrix(self, copy=True):
        transform_mat = np.eye(4, dtype=self._translation.dtype)
        transform_mat[:3, :3] = Rotation(self._quaternion).matrix
        transform_mat[3, :3] = self._translation
        return transform_mat

    @property
    def matrix(self):
        return self.get_matrix()


class TransformMatrix(object):
    """A rigid transformation represented by a 4x4 matrix"""

    @staticmethod
    def from_translation_rotation_matrix(translation, rotation_mat):
        translation = np.asarray(translation)
        rotation_mat = np.asarray(rotation_mat)
        transformation_mat = np.eye(4, dtype=translation.dtype)
        transformation_mat[:3, 3] = translation
        transformation_mat[:3, :3] = rotation_mat
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_translation_quaternion(translation, quat):
        translation = np.asarray(translation)
        quat = np.asarray(quat)
        transformation_mat = np.eye(4, dtype=translation.dtype)
        transformation_mat[:3, 3] = translation
        transformation_mat[:3, :3] = Rotation(quat).matrix
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_translation(translation):
        translation = np.asarray(translation)
        transformation_mat = np.eye(4, dtype=translation.dtype)
        transformation_mat[:3, 3] = translation
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_rotation_matrix(rotation_mat):
        rotation_mat = np.asarray(rotation_mat)
        transformation_mat = np.eye(4, dtype=rotation_mat.dtype)
        transformation_mat[:3, :3] = rotation_mat
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_quaternion(quat):
        quat = np.asarray(quat)
        transformation_mat = np.eye(4, dtype=quat.dtype)
        transformation_mat[:3, :3] = Rotation(quat).matrix
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_axis_rotation(angle, axis, point=None):
        transformation_mat = transformations.rotation_matrix(angle, axis, point)
        return TransformMatrix(transformation_mat)

    @staticmethod
    def from_x_rotation(angle):
        return TransformMatrix.axis_rotation(angle, axis=[1, 0, 0])

    @staticmethod
    def from_y_rotation(angle):
        return TransformMatrix.axis_rotation(angle, axis=[0, 1, 0])

    @staticmethod
    def from_z_rotation(angle):
        return TransformMatrix.axis_rotation(angle, axis=[0, 0, 1])

    @staticmethod
    def identity(dtype=np.float):
        return TransformMatrix(np.eye(4), dtype=dtype)

    def __init__(self, transformation_mat, copy=False):
        if copy:
            transformation_mat = np.copy(transformation_mat)
        self._transformation_mat = transformation_mat

    def get_inverse(self):
        inverse_trans = TransformMatrix(self._transformation_mat)
        inverse_trans.invert()
        return inverse_trans

    def invert(self):
        self._transformation_mat = math_utils.invert_matrix(self._transformation_mat)

    def apply_to(self, transform):
        assert isinstance(transform, TransformMatrix)
        new_transformation_mat = np.dot(self._transformation_mat, transform.get_matrix(copy=False))
        return TransformMatrix(new_transformation_mat)

    def apply_to_vector(self, vec):
        if len(vec) == 3:
            new_vec = np.dot(self._transformation_mat[:3, :3], vec)
            new_vec += self._transformation_mat[3, :3]
        elif len(vec) == 4:
            new_vec = np.dot(self._transformation_mat, vec)
        else:
            raise ValueError("Vector has to be length 3 or 4")
        return new_vec

    def get_translation(self, copy=True):
        if copy:
            return np.copy(self._transformation_mat[3, :3])
        else:
            return self._transformation_mat[3, :3]

    @property
    def translation(self):
        return self.get_translation()

    def get_rotation_matrix(self, copy=True):
        if copy:
            return np.copy(self._transformation_mat[:3, :3])
        else:
            return self._transformation_mat[:3, :3]

    @property
    def rotation_matrix(self):
        return self.get_rotation_matrix()

    def get_quaternion(self, copy=True):
        return RotationMatrix(self._transformation_mat).quaternion

    @property
    def quaternion(self):
        return self.get_quaternion()

    def get_matrix(self, copy=True):
        if copy:
            return np.copy(self._transformation_mat)
        else:
            return self._transformation_mat

    @property
    def matrix(self):
        return self.get_matrix()
