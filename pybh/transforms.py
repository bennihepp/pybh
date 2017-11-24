import numpy as np
from .contrib import transformations
from . import math_utils


class RotationMatrix(object):
    """A rotation represented by a matrix"""

    @staticmethod
    def identity():
        return RotationMatrix(np.eye(4))

    @staticmethod
    def axis_rotation(angle, axis, point=None):
        return RotationMatrix(transformations.rotation_matrix(angle, axis, point))

    @staticmethod
    def x_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[1, 0, 0])

    @staticmethod
    def y_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[0, 1, 0])

    @staticmethod
    def z_rotation(angle):
        return RotationMatrix.axis_rotation(angle, axis=[0, 0, 1])

    def __init__(self, mat=None):
        if mat is None:
            mat = RotationMatrix.identity().matrix
        elif mat.shape == (3, 3):
            new_mat = RotationMatrix.identity().matrix
            new_mat[:3, :3] = mat
            mat = new_mat
        assert(mat.shape == (4, 4))
        self._mat = mat

    def apply_to(self, transform):
        return RotationMatrix(np.dot(self._mat, transform.matrix))

    def apply_to_vector(self, vec):
        if len(vec) == 3:
            hom_vec = np.concatenate((vec, [1]))
            hom_vec = self.apply_to_vector(hom_vec)
            hom_vec /= hom_vec[3]
            return np.array(hom_vec[:3])
        else:
            vec = np.dot(self._mat, vec)
            return vec

    def get_inverse(self):
        return RotationMatrix(np.linalg.inv(self._mat))

    def invert(self):
        self._mat = np.linalg.inv(self._mat)

    def get_transpose(self):
        return RotationMatrix(np.transpose(self._mat))

    def transpose(self):
        self._mat = np.transpose(self._mat)

    @property
    def matrix(self):
        return self._mat

    @property
    def quaternion(self):
        return transformations.quaternion_from_matrix(self._mat)


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

    def __init__(self, quaternion=None):
        if quaternion is None:
            quaternion = Rotation.identity().quaternion
        self._quaternion = quaternion

    def apply_to(self, transform):
        return Rotation(math_utils.multiply_quaternion(self._quaternion, transform.quaternion))

    def apply_to_quat(self, quat):
        return math_utils.multiply_quaternion(self._quaternion, quat)

    def apply_to_vector(self, vec):
        assert(len(vec) == 3)
        return math_utils.rotate_vector_with_quaternion(self._quaternion, vec)

    def get_inverse(self):
        return Rotation(math_utils.invert_quaternion(self._quaternion))

    def invert(self):
        self._quaternion = math_utils.invert_quaternion(self._quaternion)

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def matrix(self):
        return transformations.quaternion_matrix(self._quaternion)


class Transform(object):

    def __init__(self, translation, quaternion):
        self._translation = translation
        self._quaternion = quaternion

    def get_inverse(self):
        inverse_trans = Transform(self.translation, self.quaternion)
        inverse_trans.invert()
        return inverse_trans

    def invert(self):
        self._quaternion = math_utils.invert_quaternion(self._quaternion)
        self._translation = - math_utils.rotate_vector_with_quaternion(self._quaternion, self._translation)

    @property
    def translation(self):
        return self._translation

    @property
    def quaternion(self):
        return self._quaternion
