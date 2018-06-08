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
from . import scene
from . import geometry


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
        fov = fov_degrees * np.pi / 180.0
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._update_projection_matrix()
        version_code = int("{:d}{:d}0".format(config.major_version, config.minor_version))
        self._ctx = moderngl.create_context(version_code)
        self._trackball = opengl_camera.TrackballCamera(self.width, self.height)
        self.set_trackball(self._trackball)

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
                    self._trackball.move(dx / 50., dy / 50.)

    @property
    def trackball(self):
        return self._trackball

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

    def __init__(self, background_color=None, gl_flags=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window.on_key_press = self._on_key_press
        if background_color is None:
            background_color = [0.3, 0.3, 0.3, 1.0]
        self._background_color = background_color
        self._scenes_graphs = []
        self._axis_drawer = drawing_helpers.AxisDrawer()
        self._rate = utils.RateTimer()
        if gl_flags is None:
            gl_flags = {}
        self._gl_flags = gl_flags
        self.set_gl_flag(moderngl.DEPTH_TEST, True)
        self.set_gl_flag(moderngl.CULL_FACE, True)

    def get_gl_flag(self, flag):
        return self._gl_flags.get(flag, None)

    def set_gl_flag(self, flag, enabled):
        self._gl_flags[flag] = enabled

    def clear_gl_flag(self, flag):
        del self._gl_flags[flag]

    def _iterate_scene_nodes(self):
        for graph in self._scenes_graphs:
            for node in graph.iterate_depth_first():
                yield node

    def _iterate_scene_geometries(self):
        for node in self._iterate_scene_nodes():
            if isinstance(node, scene.GeometryNode):
                yield node.geometry

    def _iterate_scene_texture_geometries(self):
        for geom in self._iterate_scene_geometries():
            if isinstance(geom, geometry.TextureTriangles3D):
                yield geom
            elif isinstance(geom, geometry.TextureMesh):
                yield geom

    def _apply_to_texture_geometries(self, func):
        for geom in self._iterate_scene_texture_geometries():
            func(geom)

    def _on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self._window.dispatch_event('on_close')
            return True
        elif symbol == key.W:
            self._ctx.wireframe = not self._ctx.wireframe
            return True
        elif symbol == key.D:
            if modifiers & key.MOD_SHIFT:
                self.set_gl_flag(moderngl.DEPTH_TEST, True)
            elif modifiers & key.MOD_CTRL:
                self.set_gl_flag(moderngl.DEPTH_TEST, False)
        elif symbol == key.F:
            if modifiers & key.MOD_SHIFT:
                self.set_gl_flag(moderngl.CULL_FACE, True)
            elif modifiers & key.MOD_CTRL:
                self.set_gl_flag(moderngl.CULL_FACE, False)
        elif symbol == key.P:
            if modifiers & key.MOD_SHIFT:
                self._apply_to_texture_geometries(lambda geom: geom.set_render_phong_shading(True))
            elif modifiers & key.MOD_CTRL:
                self._apply_to_texture_geometries(lambda geom: geom.set_render_phong_shading(False))
        elif symbol == key.N:
            if modifiers & key.MOD_SHIFT:
                self._apply_to_texture_geometries(lambda geom: geom.set_render_camera_space_normal(True))
            elif modifiers & key.MOD_CTRL:
                self._apply_to_texture_geometries(lambda geom: geom.set_render_camera_space_normal(False))
            else:
                self._apply_to_texture_geometries(lambda geom: geom.set_shader_mode(geometry.TextureMesh.SHADER_MODE_NORMAL))
            return True
        elif symbol == key.R:
            self._apply_to_texture_geometries(lambda geom: geom.set_shader_mode(geometry.TextureMesh.SHADER_MODE_NORMAL_RAW))
            return True
        elif symbol == key.T:
            self._apply_to_texture_geometries(lambda geom: geom.set_shader_mode(geometry.TextureMesh.SHADER_MODE_TEXTURE))
            return True
        elif symbol == key.U:
            self._apply_to_texture_geometries(lambda geom: geom.set_shader_mode(geometry.TextureMesh.SHADER_MODE_UNIFORM))
            return True
        elif symbol == key.C:
            self._apply_to_texture_geometries(lambda geom: geom.set_shader_mode(geometry.TextureMesh.SHADER_MODE_COLOR))
            return True

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

    def render(self, camera=None, show_fps=True, draw_axis=True, process_events=True, flip=True):
        if camera is None:
            camera = self._trackball
        for flag, enabled in self._gl_flags.items():
            if enabled:
                self._ctx.enable(flag)
            else:
                self._ctx.disable(flag)
        self._ctx.viewport = (0, 0, self._window.width, self._window.height)
        self._ctx.clear(*self._background_color)
        for graph in self._scenes_graphs:
            self._trackball.render(self._ctx, graph)

        if show_fps:
            self._rate.update_and_print_rate("FPS", print_interval=50)
        if draw_axis:
            self._axis_drawer.render(self._ctx, camera.projection_matrix, camera.view_matrix)
        if process_events:
            self.process_events()
        if flip:
            self.flip()
