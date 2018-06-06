import numpy as np
import moderngl
# from OpenGL import GL as gl
from .moderngl_utils import PointSizeScope, LineWidthScope, LazyTexture


class Object3D(object):

    def __init__(self, vertex_shader_src, fragment_shader_src, dtype=np.float32):
        self._vertex_shader_src = vertex_shader_src
        self._fragment_shader_src = fragment_shader_src
        self._initialized = False
        self._prog = None

    def _initialize_shaders(self, ctx):
        self._prog = ctx.program(vertex_shader=self._vertex_shader_src,
                                 fragment_shader=self._fragment_shader_src)

    def _initialize(self, ctx):
        raise NotImplementedError()

    def _ensure_initialized(self, ctx):
        if not self._initialized:
            self._initialize_shaders(ctx)
            self._initialize(ctx)
            self._initialized = True

    def _set_uniform(self, name, value):
        if name in self._prog._members:
            self._prog[name].value = value

    def _prepare_render(self, ctx, projection_mat, view_mat, model_mat):
        vm_mat = np.dot(view_mat, model_mat)
        pvm_mat = np.dot(projection_mat, vm_mat)
        self._set_uniform('u_projection_mat', tuple(projection_mat.T.ravel()))
        self._set_uniform('u_view_mat', tuple(view_mat.T.ravel()))
        self._set_uniform('u_model_mat', tuple(model_mat.T.ravel()))
        if 'u_normal_mat' in self._prog._members:
            normal_mat = np.linalg.inv(vm_mat[:3, :3]).T
            self._set_uniform('u_normal_mat', tuple(normal_mat.T.ravel()))
        self._set_uniform('u_vm_mat', tuple(vm_mat.T.ravel()))
        self._set_uniform('u_pvm_mat', tuple(pvm_mat.T.ravel()))

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        raise NotImplementedError()

    def render(self, ctx, projection_mat, view_mat, model_mat=None):
        if model_mat is None:
            model_mat = np.eye(4)
        self._ensure_initialized(ctx)
        self._prepare_render(ctx, projection_mat, view_mat, model_mat)
        self._render(ctx, projection_mat, view_mat, model_mat)


class UniformVertices3D(Object3D):

    VERTEX_SHADER_SRC = """
        #version 330

        in vec3 in_vert;

        uniform vec4 u_color;
        uniform mat4 u_projection_mat;
        uniform mat4 u_view_mat;
        uniform mat4 u_model_mat;

        out vec4 v_color;

        void main() {
            v_color = u_color;
            mat4 pvm = u_projection_mat * u_view_mat * u_model_mat;
            gl_Position = pvm * vec4(in_vert, 1.0);
        }
    """

    FRAGMENT_SHADER_SRC = """
        #version 330

        in vec4 v_color;

        out vec4 f_color;

        void main() {
            f_color = v_color;
        }
    """

    def __init__(self, vertices, render_mode, color, indices=None, dtype=np.float32, int_dtype=np.int32):
        super().__init__(self.VERTEX_SHADER_SRC, self.FRAGMENT_SHADER_SRC, dtype)
        self._vertices = np.ascontiguousarray(vertices, dtype=dtype)
        self._render_mode = render_mode
        self._color = color
        self._indices = indices
        if indices is None:
            self._num_primitives = self._vertices.size // 3
        else:
            self._indices = np.ascontiguousarray(self._indices, dtype=int_dtype)
            self._num_primitives = self._indices.size
        self._render_mode = render_mode
        self._vbo = None
        self._vao = None
        self._ibo = None

    def override_vertex_shader(self, vertex_shader_src):
        assert not self._initialized
        self._vertex_shader_src = vertex_shader_src

    def override_fragment_shader(self, fragment_shader_src):
        assert not self._initialized
        self._fragment_shader_src = fragment_shader_src

    def set_color(self, color):
        self._color = np.ascontiguousarray(color, dtype=self._vertices.dtype)

    def _initialize(self, ctx):
        self._vbo = ctx.buffer(self._vertices)
        vao_content = [
            (self._vbo, '3f', 'in_vert'),
        ]
        if self._indices is None:
            self._vao = ctx.vertex_array(self._prog, vao_content)
            self._num_primitives = self._vertices.size // 3
        else:
            self._ibo = ctx.buffer(self._indices)
            self._vao = ctx.vertex_array(self._prog, vao_content, self._ibo)
            self._num_primitives = self._indices.size

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        self._set_uniform('u_color', tuple(self._color))
        self._vao.render(self._render_mode, len(self._vertices))


class UniformPoints3D(UniformVertices3D):

    def __init__(self, points, color, point_size=1.0, indices=None, dtype=np.float32):
        super().__init__(points, moderngl.POINTS, color, indices, dtype)
        self._point_size_scope = PointSizeScope(point_size)

    def set_point_size(self, point_size):
        self._point_size_scope.set_point_size(point_size)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._point_size_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class UniformLines3D(UniformVertices3D):

    def __init__(self, lines, color, line_width=1.0, indices=None, dtype=np.float32):
        super().__init__(lines, moderngl.LINES, color, indices, dtype)
        self._line_width_scope = LineWidthScope(line_width)

    def set_line_width(self, line_width):
        self._line_width_scope.set_line_width(line_width)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._line_width_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class UniformLineStrip3D(UniformVertices3D):

    def __init__(self, lines, color, line_width=1.0, indices=None, dtype=np.float32):
        super().__init__(lines, moderngl.LINE_STRIP, color, indices, dtype)
        self._line_width_scope = LineWidthScope(line_width)

    def set_line_width(self, line_width):
        self._line_width_scope.set_line_width(line_width)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._line_width_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class UniformTriangles3D(UniformVertices3D):

    def __init__(self, vertices, colors, indices=None, dtype=np.float32):
        super().__init__(vertices, colors, moderngl.TRIANGLES, indices, dtype)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        super()._render(ctx, projection_mat, view_mat, model_mat)


class Vertices3D(Object3D):

    VERTEX_SHADER_SRC = """
        #version 330

        in vec3 in_vert;
        in vec4 in_color;

        uniform mat4 u_projection_mat;
        uniform mat4 u_view_mat;
        uniform mat4 u_model_mat;

        out vec4 v_color;

        void main() {
            v_color = in_color;
            mat4 pvm = u_projection_mat * u_view_mat * u_model_mat;
            gl_Position = pvm * vec4(in_vert, 1.0);
        }
    """

    FRAGMENT_SHADER_SRC = """
        #version 330

        in vec4 v_color;

        out vec4 f_color;

        void main() {
            f_color = v_color;
        }
    """

    def __init__(self, vertices, colors, render_mode, indices=None, dtype=np.float32, int_dtype=np.int32):
        super().__init__(self.VERTEX_SHADER_SRC, self.FRAGMENT_SHADER_SRC, dtype)
        assert len(vertices) == len(colors)
        self._vertices = np.concatenate([vertices, colors], axis=1)
        self._vertices = np.ascontiguousarray(self._vertices, dtype=dtype)
        self._indices = indices
        if indices is None:
            self._num_primitives = self._vertices.size // 7
        else:
            self._indices = np.ascontiguousarray(self._indices, dtype=int_dtype)
            self._num_primitives = self._indices.size
        self._render_mode = render_mode
        self._vbo = None
        self._vao = None
        self._ibo = None

    def override_vertex_shader(self, vertex_shader_src):
        assert not self._initialized
        self._vertex_shader_src = vertex_shader_src

    def override_fragment_shader(self, fragment_shader_src):
        assert not self._initialized
        self._fragment_shader_src = fragment_shader_src

    def _initialize(self, ctx):
        self._vbo = ctx.buffer(self._vertices)
        vao_content = [
            (self._vbo, '3f 4f', 'in_vert', 'in_color'),
        ]
        if self._indices is None:
            self._vao = ctx.vertex_array(self._prog, vao_content)
        else:
            self._ibo = ctx.buffer(self._indices)
            self._vao = ctx.vertex_array(self._prog, vao_content, self._ibo)
        # self._vao = ctx.simple_vertex_array(self._prog, self._vbo, 'in_vert', 'in_color')

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        self._vao.render(self._render_mode, self._num_primitives)


class Points3D(Vertices3D):

    def __init__(self, points, colors, point_size=1.0, indices=None, dtype=np.float32):
        super().__init__(points, colors, moderngl.POINTS, indices, dtype)
        self._point_size_scope = PointSizeScope(point_size)

    def set_point_size(self, point_size):
        self._point_size_scope.set_point_size(point_size)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._point_size_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class Lines3D(Vertices3D):

    def __init__(self, lines, colors, line_width=1.0, indices=None, dtype=np.float32):
        super().__init__(lines, colors, moderngl.LINES, indices, dtype)
        self._line_width_scope = LineWidthScope(line_width)

    def set_line_width(self, line_width):
        self._line_width_scope.set_line_width(line_width)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._line_width_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class Axis3D(Lines3D):

    def __init__(self, scale=1.0, line_width=1.0, dtype=np.float32):
        lines = [
            scale * np.array([0, 0, 0]),
            scale * np.array([1, 0, 0]),
            scale * np.array([0, 0, 0]),
            scale * np.array([0, 1, 0]),
            scale * np.array([0, 0, 0]),
            scale * np.array([0, 0, 1]),
        ]
        colors = [
            np.array([1, 0, 0, 1]),
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
        ]

        super().__init__(lines, colors, line_width=line_width, dtype=dtype)


class LineStrip3D(Vertices3D):

    def __init__(self, lines, colors, line_width=1.0, indices=None, dtype=np.float32):
        super().__init__(lines, colors, moderngl.LINE_STRIP, indices, dtype)
        self._line_width_scope = LineWidthScope(line_width)

    def set_line_width(self, line_width):
        self._line_width_scope.set_line_width(line_width)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        with self._line_width_scope.use(ctx):
            super()._render(ctx, projection_mat, view_mat, model_mat)


class TrianglesData(object):

    def _compute_normals(self, vertices, indices, dtype):
        if indices is None:
            indices = np.arange(len(vertices))
        indices = np.reshape(indices, [-1, 3])
        face_normals = []
        for face_indices in indices:
            v1 = vertices[face_indices[0]]
            v2 = vertices[face_indices[1]]
            v3 = vertices[face_indices[2]]
            normal = np.cross(v2 - v1, v3 - v2)
            normal /= np.linalg.norm(normal)
            face_normals.append(normal)
        vertex_normals = np.zeros((len(vertices), 3), dtype=dtype)
        vertex_counts = np.zeros((len(vertices)), dtype=np.int)
        for face_id, face_indices in enumerate(indices):
            for vert_index in face_indices:
                normal = face_normals[face_id]
                vertex_counts[vert_index] += 1
                vertex_normals[vert_index] += normal
        vertex_mask = vertex_counts > 0
        vertex_normals[vertex_mask, ...] /= vertex_counts[vertex_mask, np.newaxis]
        vertex_normals[vertex_mask, ...] /= np.linalg.norm(vertex_normals[vertex_mask, ...], axis=1)[..., np.newaxis]
        return vertex_normals

    def __init__(self, vertices, colors, normals=None, indices=None,
                 dtype=np.float32, int_dtype=np.int32,
                 compute_normals_if_invalid=True):
        if colors is None:
            colors = np.ones((len(vertices), 4), dtype=dtype)
            colors[..., :3] *= 0.5
        assert len(vertices) == len(colors)
        self._vertices = np.ascontiguousarray(vertices, dtype=dtype)
        self._colors = np.ascontiguousarray(colors, dtype=dtype)
        if normals is None:
            if compute_normals_if_invalid:
                self._normals = self._compute_normals(self._vertices, indices, dtype)
            else:
                self._normals = None
        else:
            assert len(vertices) == len(normals)
            self._normals = np.ascontiguousarray(normals, dtype=dtype)
            if compute_normals_if_invalid and np.any(np.logical_not(np.isfinite(self._normals))):
                self._normals = self._compute_normals(self._vertices, indices, dtype)
        if indices is not None:
            self._indices = np.ascontiguousarray(indices, dtype=int_dtype)
        else:
            self._indices = None

    @property
    def vertices(self):
        return self._vertices

    @property
    def colors(self):
        return self._colors

    def has_normals(self):
        return len(self._normals) > 0

    @property
    def normals(self):
        return self._normals

    @property
    def indices(self):
        return self._indices


class Triangles3D(Vertices3D):

    @staticmethod
    def from_triangles_data(triangles_data):
        return Triangles3D(triangles_data.vertices, triangles_data.colors,
                           triangles_data.indices, triangles_data.vertices.dtype)

    def __init__(self, vertices, colors, indices=None, dtype=np.float32):
        super().__init__(vertices, colors, moderngl.TRIANGLES, indices, dtype)

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        super()._render(ctx, projection_mat, view_mat, model_mat)


class TextureTrianglesData(object):

    def _compute_normals(self, vertices, indices, dtype):
        if indices is None:
            indices = np.arange(len(vertices))
        indices = np.reshape(indices, [-1, 3])
        face_normals = []
        for face_indices in indices:
            v1 = vertices[face_indices[0]]
            v2 = vertices[face_indices[1]]
            v3 = vertices[face_indices[2]]
            normal = np.cross(v2 - v1, v3 - v2)
            normal /= np.linalg.norm(normal)
            face_normals.append(normal)
        vertex_normals = np.zeros((len(vertices), 3), dtype=dtype)
        vertex_counts = np.zeros((len(vertices)), dtype=np.int)
        for face_id, face_indices in enumerate(indices):
            for vert_index in face_indices:
                normal = face_normals[face_id]
                vertex_counts[vert_index] += 1
                vertex_normals[vert_index] += normal
        vertex_mask = vertex_counts > 0
        vertex_normals[vertex_mask, ...] /= vertex_counts[vertex_mask, np.newaxis]
        vertex_normals[vertex_mask, ...] /= np.linalg.norm(vertex_normals[vertex_mask, ...], axis=1)[..., np.newaxis]
        return vertex_normals

    def __init__(self, vertices, texcoords=None, texture=None, colors=None, normals=None, indices=None,
                 dtype=np.float32, int_dtype=np.int32, compute_normals_if_invalid=True):
        if texcoords is not None:
            assert len(vertices) == len(texcoords)
        if colors is not None:
            assert len(vertices) == len(colors)
        if normals is not None:
            assert len(vertices) == len(normals)
        self._vertices = np.ascontiguousarray(vertices, dtype=dtype)
        assert self.vertices.shape[1] == 3
        if texcoords is None:
            self._texcoords = None
        else:
            self._texcoords = np.ascontiguousarray(texcoords, dtype=dtype)
            assert self._texcoords.shape[1] == 2
        self._texture = texture
        if colors is None:
            if self._texcoords is None:
                self._colors = np.ones((len(vertices), 4), dtype=dtype)
                self._colors[..., :3] *= 0.5
            else:
                self._colors = None
        else:
            self._colors = np.ascontiguousarray(colors, dtype=dtype)
            assert self._colors.shape[1] == 4
        if normals is None:
            if compute_normals_if_invalid:
                self._normals = self._compute_normals(self._vertices, indices, dtype)
            else:
                self._normals = None
        else:
            assert len(vertices) == len(normals)
            self._normals = np.ascontiguousarray(normals, dtype=dtype)
            if compute_normals_if_invalid and np.any(np.logical_not(np.isfinite(self._normals))):
                self._normals = self._compute_normals(self._vertices, indices, dtype)
            assert self._normals.shape[1] == 3
        if indices is not None:
            self._indices = np.ascontiguousarray(indices, dtype=int_dtype)
        else:
            self._indices = None

    @property
    def vertices(self):
        return self._vertices

    @property
    def texcoords(self):
        return self._texcoords

    @property
    def texture(self):
        return self._texture

    def has_normals(self):
        return len(self._normals) > 0

    @property
    def colors(self):
        return self._colors

    @property
    def normals(self):
        return self._normals

    @property
    def indices(self):
        return self._indices


class TextureTriangles3D(Object3D):

    VERTEX_SHADER_SRC = """
        #version 330

        in vec3 in_vert;
        in vec3 in_norm;
        in vec4 in_color;
        in vec2 in_texcoord;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec4 v_color;
        out vec2 v_texcoord;

        uniform bool u_render_normal;
        uniform bool u_render_texture;
        uniform bool u_camera_space_normal;
        uniform mat4 u_pvm_mat;
        uniform mat3 u_normal_mat;

        void main() {
            gl_Position = u_pvm_mat * vec4(in_vert, 1.0);
            if (u_render_normal && u_camera_space_normal) {
                v_norm = u_normal_mat * in_norm;
            }
            else {
                v_norm = in_norm;
                if (u_render_texture) {
                    v_texcoord = in_texcoord;
                }
                else {
                    v_color = in_color;
                }
            }
        }
    """

    FRAGMENT_SHADER_SRC = """
        #version 330

        in vec3 v_vert;
        in vec3 v_norm;
        in vec4 v_color;
        in vec2 v_texcoord;

        out vec4 f_color;

        uniform bool u_render_normal;
        uniform bool u_raw_normal;
        uniform bool u_render_texture;
        uniform sampler2D u_texture;
        uniform vec4 u_color;

        void main() {
            float lum = 1;
            if (u_render_normal) {
                vec3 color;
                if (u_raw_normal) {
                    color = v_norm;
                }
                else {
                    color = 0.5 * v_norm + 0.5;
                }
                f_color = vec4(color * lum, 1.0);
            }
            else if (u_render_texture) {
                f_color = vec4(texture(u_texture, v_texcoord).rgb * lum, 1.0);
            }
            else {
                f_color = v_color * lum;
            }
        }
    """

    def __init__(self, triangles_data, color=None):
        super().__init__(self.VERTEX_SHADER_SRC, self.FRAGMENT_SHADER_SRC, triangles_data.vertices.dtype)
        assert triangles_data.has_normals()
        self._triangles_data = triangles_data
        self._texture = triangles_data.texture
        if triangles_data.indices is None:
            self._num_primitives = triangles_data.vertices.size // 3
        else:
            self._num_primitives = triangles_data.indices.size
        self._render_mode = moderngl.TRIANGLES
        self._vertex_vbo = None
        self._texcoord_vbo = None
        self._normal_vbo = None
        self._vao = None
        self._ibo = None
        if color is None:
            color = np.array([0.5, 0.5, 0.5, 1], dtype=triangles_data.vertices.dtype)
        self._color = color
        self._render_normal = False
        self._raw_normal = False
        self._camera_space_normal = False
        if self._triangles_data.texcoords is not None and self._triangles_data.texture is not None:
            self._render_texture = True
        else:
            self._render_texture = False

    @property
    def render_normal(self):
        return self._render_normal

    @render_normal.setter
    def render_normal(self, render_normal):
        self._render_normal = render_normal

    @property
    def raw_normal(self):
        return self._raw_normal

    @raw_normal.setter
    def raw_normal(self, raw_normal):
        self._raw_normal = raw_normal

    @property
    def camera_space_normal(self):
        return self._camera_space_normal

    @camera_space_normal.setter
    def camera_space_normal(self, camera_space_normal):
        self._camera_space_normal = camera_space_normal

    @property
    def render_texture(self):
        return self._render_texture

    @render_texture.setter
    def render_texture(self, render_texture):
        self._render_texture = render_texture

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    def override_vertex_shader(self, vertex_shader_src):
        assert not self._initialized
        self._vertex_shader_src = vertex_shader_src

    def override_fragment_shader(self, fragment_shader_src):
        assert not self._initialized
        self._fragment_shader_src = fragment_shader_src

    def _initialize(self, ctx):
        vao_content = []
        self._vertex_vbo = ctx.buffer(self._triangles_data.vertices)
        vao_content.append((self._vertex_vbo, '3f', 'in_vert'))
        if self._triangles_data.colors is not None and 'in_color' in self._prog._members:
            self._color_vbo = ctx.buffer(self._triangles_data.colors)
            vao_content.append((self._color_vbo, '4f', 'in_color'))
        if self._triangles_data.texcoords is not None and 'in_texcoord' in self._prog._members:
            self._texcoord_vbo = ctx.buffer(self._triangles_data.texcoords)
            vao_content.append((self._texcoord_vbo, '2f', 'in_texcoord'))
        if self._triangles_data.normals is not None and 'in_norm' in self._prog._members:
            self._normal_vbo = ctx.buffer(self._triangles_data.normals)
            vao_content.append((self._normal_vbo, '3f', 'in_norm'))
        if self._triangles_data.indices is not None:
            self._ibo = ctx.buffer(self._triangles_data.indices)
        self._vao = ctx.vertex_array(self._prog, vao_content, self._ibo)
        if self._texture is not None:
            if isinstance(self._texture, LazyTexture):
                self._texture.ensure_initialized(ctx)
                self._texture.build_mipmaps()
            elif not isinstance(self._texture, moderngl.Texture):
                self._texture = np.ascontiguousarray(self._texture)
                assert self._texture.dtype == np.uint8
                height, width, channels = self._texture.shape
                assert channels == 3
                self._texture = ctx.texture((width, height), 3, self._texture)
                self._texture.build_mipmaps()

    def _render(self, ctx, projection_mat, view_mat, model_mat):
        self._set_uniform('u_render_normal', self._render_normal)
        self._set_uniform('u_raw_normal', self._raw_normal)
        self._set_uniform('u_camera_space_normal', self._camera_space_normal)
        if 'u_color' in self._prog._members:
            self._set_uniform('u_color', tuple(self._color))
        if self._texture is None:
            self._set_uniform('u_render_texture', False)
        else:
            self._set_uniform('u_render_texture', self._render_texture)
            self._texture.use()
        self._vao.render(self._render_mode, self._num_primitives)


class TextureMesh(object):

    def __init__(self, mesh_data, color=None):
        if isinstance(mesh_data, TextureTrianglesData):
            mesh_data = [mesh_data]
        assert len(mesh_data) > 0
        self._triangles_list = []
        for triangles_data in mesh_data:
            if isinstance(triangles_data, TextureTrianglesData):
                triangles = TextureTriangles3D(triangles_data, color)
            else:
                raise ValueError("Unsupported triangle data object")
            self._triangles_list.append(triangles)

    def render(self, ctx, projection_mat, view_mat, model_mat=None):
        for triangles in self._triangles_list:
            triangles.render(ctx, projection_mat, view_mat, model_mat)

    @property
    def render_normal(self):
        return self._triangles_list[0].render_normal

    @render_normal.setter
    def render_normal(self, render_normal):
        for triangles in self._triangles_list:
            triangles.render_normal = render_normal

    @property
    def raw_normal(self):
        return self._triangles_list[0].raw_normal

    @raw_normal.setter
    def raw_normal(self, raw_normal):
        for triangles in self._triangles_list:
            triangles.raw_normal = raw_normal

    @property
    def camera_space_normal(self):
        return self._triangles_list[0].camera_space_normal

    @camera_space_normal.setter
    def camera_space_normal(self, camera_space_normal):
        for triangles in self._triangles_list:
            triangles.camera_space_normal = camera_space_normal

    @property
    def render_texture(self):
        return self._triangles_list[0].render_texture

    @render_texture.setter
    def render_texture(self, render_texture):
        for triangles in self._triangles_list:
            triangles.render_texture = render_texture

    @property
    def color(self):
        return self._triangles_list[0].color

    @color.setter
    def color(self, color):
        for triangles in self._triangles_list:
            triangles.color = color
