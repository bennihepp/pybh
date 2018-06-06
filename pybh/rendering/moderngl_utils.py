import numpy as np
import contextlib
from PIL import Image


class LazyTexture(object):

    @staticmethod
    def from_file(filename, flip_top_bottom=True, verbose=True):
        image = Image.open(filename).convert('RGB')
        if flip_top_bottom:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_data = np.asarray(image)
        print("Loaded texture from {:s}. Shape: {}, Dtype: {}".format(filename, image_data.shape, image_data.dtype))
        return LazyTexture(image_data)

    def __init__(self, image_data):
        self._image_data = np.asarray(image_data)
        assert self._image_data.dtype == np.uint8
        self._texture = None

    def ensure_initialized(self, ctx):
        self._texture = np.asarray(self._image_data)
        height, width, channels = self._texture.shape
        assert channels == 3
        self._texture = ctx.texture((width, height), 3, self._texture)

    def __getattr__(self, item):
        return self._texture.__getattribute__(item)

    @property
    def texture(self):
        return self._texture

    @property
    def image_data(self):
        return self._image_data


class PointSizeScope(object):

    # def __init__(self, point_size):
    #     self._point_size = point_size
    #     self._prev_point_size = gl.GLfloat()
    #
    # def __enter__(self):
    #     gl.glGetFloatv(gl.GL_POINT_SIZE, self._prev_point_size)
    #     gl.glPointSize(self._point_size)
    #
    # def __exit__(self, type, value, traceback):
    #     gl.glPointSize(self._prev_point_size)
    #
    # def set_point_size(self, point_size):
    #     self._point_size = point_size

    def __init__(self, point_size):
        self._point_size = point_size
        self._prev_point_size = None


    @contextlib.contextmanager
    def use(self, ctx):
        self._prev_point_size = ctx.point_size
        ctx.point_size = self._point_size
        yield
        ctx.point_size = self._prev_point_size

    def set_point_size(self, point_size):
        self._point_size = point_size


class LineWidthScope(object):

    # def __init__(self, line_width):
    #     self._line_width = line_width
    #     self._prev_line_width = gl.GLfloat()
    #
    # def __enter__(self):
    #     gl.glGetFloatv(gl.GL_LINE_WIDTH, self._prev_line_width)
    #     gl.glLineWidth(self._line_width)
    #
    # def __exit__(self, type, value, traceback):
    #     gl.glLineWidth(self._prev_line_width)

    # def set_line_width(self, line_width):
    #     self._line_width = line_width

    def __init__(self, line_width):
        self._line_width = line_width
        self._prev_line_width = None

    @contextlib.contextmanager
    def use(self, ctx):
        self._prev_line_width = ctx.line_width
        ctx.line_width = self._line_width
        yield
        ctx.line_width = self._prev_line_width

    def set_line_width(self, line_width):
        self._line_width = line_width
