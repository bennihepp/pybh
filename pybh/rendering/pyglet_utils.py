import sys
import numpy as np
import moderngl
import pyglet
if sys.platform == 'darwin':
    # Prevent creation of shadow window with wrong OpenGL context version
    pyglet.options['shadow_window'] = False
from pyglet import gl
from pyglet.window import key
from . import opengl_camera
from . import opengl_math
from . import drawing_helpers
from .. import utils


class PygletViewer(object):

    def __init__(self, width=800, height=600,
                 fov_degrees=90., znear=0.1, zfar=1000.0,
                 visible=True, vsync=True, resizable=True,
                 config=None):
        if config is None:
            config = gl.Config(double_buffer=True, depth_size=24, major_version=4, minor_version=1)

        self._window = pyglet.window.Window(width=width, height=height, visible=visible, vsync=vsync,
                                            resizable=resizable, config=config)
        self._window.on_resize = self._on_resize
        self._window.on_key_press = self._on_key_press
        self._window.on_mouse_press = self._on_mouse_press
        self._window.on_mouse_drag = self._on_mouse_drag
        self._window.on_mouse_scroll = self._on_mouse_scroll
        self._cameras = []
        self._trackball = None
        fov = fov_degrees * np.pi / 180.0
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._update_projection_matrix()
        version_code = int("{:d}{:d}0".format(config.major_version, config.minor_version))
        self._ctx = moderngl.create_context(version_code)

    def _update_projection_matrix(self):
        self._projection_matrix = opengl_math.perspective_matrix(
            self._fov, self._window.width / self._window.height, self._znear, self._zfar)

    def _on_resize(self, width, height):
        self._update_projection_matrix()
        for camera in self._cameras:
            camera.set_size(width, height)
        if self._trackball is not None:
            self._trackball.set_size(width, height)

    def _on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self._window.dispatch_event('on_close')
            return True

    def _on_mouse_press(self, x, y, button, modifiers):
        if self._trackball is not None:
            pass

    def _on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self._trackball is not None:
            self._trackball.zoom(scroll_y / 5.)

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._trackball is not None:
            if buttons == 1:
                if modifiers == 0:
                    self._trackball.rotate(x, y, dx / 100., dy / 100.)
                elif modifiers == 1:
                    self._trackball.move(dx / 10., dy / 10.)

    @property
    def ctx(self):
        return self._ctx

    @property
    def projection_matrix(self):
        return self._projection_matrix

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

    @fov.setter
    def znear(self, znear):
        self._znear = znear
        self._update_projection_matrix()

    @property
    def zfar(self):
        return self._zfar

    @fov.setter
    def zfar(self, zfar):
        self._zfar = zfar
        self._update_projection_matrix()

    @property
    def window(self):
        return self._window

    @property
    def should_quit(self):
        return self._window.has_exit

    @property
    def width(self):
        return self._window.width

    @property
    def height(self):
        return self._window.height

    def process_events(self):
        return self._window.dispatch_events()

    def flip(self):
        return self._window.flip()

    def register_camera(self, camera):
        self._cameras.append(camera)
        camera.set_size(self.width, self.height)

    def unregister_camera(self, camera):
        self._cameras.remove(camera)

    def set_trackball(self, trackball):
        self._trackball = trackball
        if self._trackball is not None:
            self._trackball.set_size(self._window.width, self._window.height)


class PygletSceneViewer(PygletViewer):

    def __init__(self, background_color=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window.on_key_press = self._on_key_press
        if background_color is None:
            background_color = [0.3, 0.3, 0.3, 1.0]
        self._background_color = background_color
        self._scenes_graphs = []
        trackball = opengl_camera.TrackballCamera(self.width, self.height)
        self.set_trackball(trackball)
        self._axis_drawer = drawing_helpers.AxisDrawer(trackball)
        self._rate = utils.RateTimer()

    def _on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self._window.dispatch_event('on_close')
            return True
        elif symbol == key.W:
            self._ctx.wireframe = not self._ctx.wireframe
            return True

    @property
    def camera(self):
        return self._trackball

    def get_background_color(self):
        return self._background_color

    def set_background_color(self, background_color):
        self._background_color = background_color

    def get_scenes(self):
        return list(self._scenes_graphs)

    def add_scene(self, scene):
        self._scenes_graphs.append(scene)

    def remove_scene(self, scene):
        self._scenes_graphs.remove(scene)

    def remove_scene_at_index(self, i):
        assert i >= 0
        assert i < len(self._scenes_graphs)
        del self._scenes_graphs[i]

    def render(self, show_fps=True, draw_axis=True, process_events=True, flip=True):
        self._ctx.viewport = (0, 0, self._window.width, self._window.height)
        self._ctx.clear(*self._background_color)
        for graph in self._scenes_graphs:
            self._trackball.render(self._ctx, graph)

        if show_fps:
            self._rate.update_and_print_rate("FPS", print_interval=50)
        if draw_axis:
            self._axis_drawer.render(self._ctx)
        if process_events:
            self.process_events()
        if flip:
            self.flip()
