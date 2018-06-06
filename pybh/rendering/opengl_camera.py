import numpy as np
from .. import transforms
from . import opengl_math


class Camera(object):

    def __init__(self, width, height, fov_degrees=90., znear=0.1, zfar=1000.0, dtype=np.float32):
        self._width = width
        self._height = height
        fov = fov_degrees * np.pi / 180.0
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._view_matrix = np.eye(4, dtype=dtype)
        self._update_projection_matrix()
        self.look_at([0, 0, 10], [0, 0, 0], [0, 1, 0])

    def _update_projection_matrix(self):
        self._projection_matrix = opengl_math.perspective_matrix(
            self._fov, self._width / self._height, self._znear, self._zfar)

    def set_size(self, width, height):
        self._width = width
        self._height = height
        self._update_projection_matrix()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width
        self._update_projection_matrix()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height
        self._update_projection_matrix()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov
        self._update_projection_matrix()

    @property
    def znear(self):
        return self._znear

    @znear.setter
    def znear(self, znear):
        self._znear = znear
        self._update_projection_matrix()

    @property
    def zfar(self):
        return self._zfar

    @zfar.setter
    def zfar(self, zfar):
        self._zfar = zfar
        self._update_projection_matrix()

    @property
    def projection_matrix(self):
        return self._projection_matrix

    @property
    def view_matrix(self):
        return self._view_matrix

    @property
    def view_matrix_without_translation(self):
        view_matrix = np.copy(self._view_matrix)
        view_matrix[:3, 3] = 0
        return view_matrix

    @view_matrix.setter
    def view_matrix(self, view_matrix):
        self._view_matrix = view_matrix

    def look_at(self, eye, center, up):
        self._view_matrix = opengl_math.look_at(eye, center, up, dtype=self._view_matrix.dtype)

    def render(self, ctx, object):
        object.render(ctx, self.projection_matrix, self.view_matrix)


class TrackballCamera(Camera):

    def __init__(self, width, height, fov_degrees=90., znear=0.1, zfar=1000.0, dtype=np.float32):
        super().__init__(width, height, fov_degrees, znear, zfar, dtype)
        self._zoom_translation = np.array(self._view_matrix[:3, 3])
        self._center_translation = np.zeros((3,), dtype=self._view_matrix.dtype)
        self._view_matrix[:3, 3] = 0

    def look_at(self, eye, center, up):
        self._view_matrix = opengl_math.look_at(eye, center, up, dtype=self._view_matrix.dtype)
        self._zoom_translation = np.array(self._view_matrix[:3, 3])
        self._view_matrix[:3, 3] = 0

    def zoom(self, amount):
        self._zoom_translation += [0, 0, amount]

    def move(self, dx, dy):
        dt = np.array([dx, dy, 0, 1], dtype=self._view_matrix.dtype)
        dt = np.dot(np.linalg.inv(self._view_matrix), dt)
        self._center_translation += dt[:3]

    def rotate(self, x, y, dx, dy):
        x_axis = np.array([1, 0, 0], dtype=self._view_matrix.dtype)
        y_axis = np.array([0, 1, 0], dtype=self._view_matrix.dtype)
        rot_x = transforms.TransformMatrix.from_axis_rotation(-dy, x_axis)
        rot_y = transforms.TransformMatrix.from_axis_rotation(+dx, y_axis)
        self._view_matrix = rot_x.apply_to(rot_y).apply_to(transforms.TransformMatrix(self._view_matrix)).matrix
        # self._view_matrix = np.dot(np.dot(rot_x.matrix, rot_y.matrix), self._view_matrix)

    @property
    def view_matrix(self):
        center_transform = transforms.TransformMatrix.from_translation(self._center_translation)
        zoom_transform = transforms.TransformMatrix.from_translation(self._zoom_translation)
        view_matrix = zoom_transform.apply_to(transforms.TransformMatrix(self._view_matrix)).apply_to(
            center_transform).matrix
        return view_matrix
