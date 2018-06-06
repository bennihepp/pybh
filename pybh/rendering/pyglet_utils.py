import sys
import numpy as np
import moderngl
import pyglet
if sys.platform == 'darwin':
    # Prevent creation of shadow window with wrong OpenGL context version
    pyglet.options['shadow_window'] = False
from pyglet import gl
from pyglet.window import key
from . import opengl_math


class PygletWindow(object):

    def __init__(self, width=800, height=600,
                 fov_degrees=90., znear=0.1, zfar=1000.0,
                 visible=True, vsync=True, config=None):
        if config is None:
            config = gl.Config(double_buffer=True, depth_size=24, major_version=4, minor_version=1)

        self._window = pyglet.window.Window(visible=False, resizable=True, config=config)
        self._window.on_key_press = self._on_key_press
        self._window.on_mouse_press = self._on_mouse_press
        self._window.on_mouse_drag = self._on_mouse_drag
        self._window.on_mouse_scroll = self._on_mouse_scroll
        self._window.on_resize = self._on_resize
        self._window.set_size(width, height)
        self._cameras = []
        self._trackball = None
        fov = fov_degrees * np.pi / 180.0
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._window.set_visible(visible)
        self._window.set_vsync(vsync)
        self._update_projection_matrix()
        self._ctx = moderngl.create_context()

    def register_camera(self, camera):
        self._cameras.append(camera)
        camera.set_size(self.width, self.height)

    def unregister_camera(self, camera):
        self._cameras.remove(camera)

    def set_trackball(self, trackball):
        self._trackball = trackball
        if self._trackball is not None:
            self._trackball.set_size(self.width, self.height)

    @property
    def window(self):
        return self._window

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
        elif symbol == key.W:
            self._ctx.wireframe = not self._ctx.wireframe
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

    def __getattr__(self, item):
        return self._window.__getattribute__(item)
