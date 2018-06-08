import numpy as np
import moderngl
from . import geometry
from .. import transforms


class AxisDrawer(object):

    def __init__(self, scale=8.0, line_width=5.0, viewport_fraction=1/4, background_color=None):
        self._axis = geometry.Axis3D(scale=scale, line_width=line_width)
        self._viewport_fraction = viewport_fraction
        if background_color is None:
            background_color = [0.3, 0.3, 0.3, 1.0]
        self._background_color = background_color

    def render(self, ctx, projection_mat, view_mat, model_mat=None, clear=False):
        old_viewport = ctx.viewport
        x1, y1, x2, y2 = old_viewport
        w = x2 - x1
        h = y2 - y1
        new_w = self._viewport_fraction * w
        new_h = self._viewport_fraction * h
        new_viewport = (x1, y1, int(new_w), int(new_h))
        ctx.viewport = new_viewport
        if clear:
            ctx.clear(*self._background_color, viewport=ctx.viewport)
        axis_view_matrix = np.copy(view_mat)
        axis_view_matrix[:3, 3] = 0
        axis_view_matrix = np.dot(transforms.TransformMatrix.from_translation([0, 0, -10]).matrix, axis_view_matrix)
        ctx.disable(moderngl.DEPTH_TEST)
        self._axis.render(ctx, projection_mat, axis_view_matrix)
        # ctx.enable(moderngl.DEPTH_TEST)
        ctx.viewport = old_viewport
