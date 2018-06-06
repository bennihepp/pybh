import numpy as np
from pybh.contrib import transformations
from . import geometry


class SceneNode(object):

    def __init__(self, visible=True, name=None):
        if name is None:
            name = "<unnamed>"
        self._name = name
        self._children = []
        self._visible = visible

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        self._children = children

    @property
    def is_leaf(self):
        return self.children is not None and len(self.children) > 0

    def add_child(self, child_node):
        self._children.append(child_node)

    def render(self, ctx, projection_mat, view_mat, model_mat):
        if self.visible:
            for child_node in self.children:
                child_node.render(ctx, projection_mat, view_mat, model_mat)


class TransformationNode(SceneNode):

    def __init__(self, transformation):
        super().__init__()
        self._transformation = transformation

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        self._transformation = transformation

    def render(self, ctx, projection_mat, view_mat, model_mat):
        new_model_mat = model_mat.dot(self._transformation)
        for child_node in self.children:
            child_node.render(ctx, projection_mat, view_mat, new_model_mat)


class ScaleNode(TransformationNode):

    def __init__(self, scale, dtype=np.float32):
        scale = np.asarray(scale)
        transformation = scale * np.eye(4, dtype=dtype)
        transformation[3, 3] = 1
        super().__init__(transformation)


class TranslationNode(TransformationNode):

    def __init__(self, translation):
        translation = np.asarray(translation)
        transformation = np.eye(4, dtype=translation.dtype)
        transformation[:3, 3] = translation
        super().__init__(transformation)


class RotationMatrixNode(TransformationNode):

    def __init__(self, rot_mat):
        rot_mat = np.asarray(rot_mat)
        transformation = np.eye(4, dtype=rot_mat.dtype)
        transformation[:3, :3] = rot_mat
        super().__init__(transformation)


class RotationQuaternionNode(TransformationNode):

    def __init__(self, rot_quat):
        """
        Rotation based on quaternion `rot_quat = [w, x, y, z]`.
        """
        rot_quat = np.asarray(rot_quat)
        rot_mat = transformations.quaternion_matrix(rot_quat)
        transformation = rot_mat
        super().__init__(transformation)


class RotationAxisAngleNode(RotationQuaternionNode):

    def __init__(self, angle, axis):
        rot_quat = transformations.quaternion_about_axis(angle, axis)
        super().__init__(rot_quat)


class RotationXNode(RotationAxisAngleNode):

    def __init__(self, angle, dtype=np.float32):
        axis = np.array([1, 0, 0], dtype=dtype)
        super().__init__(angle, axis)


class RotationYNode(RotationAxisAngleNode):

    def __init__(self, angle, dtype=np.float32):
        axis = np.array([0, 1, 0], dtype=dtype)
        super().__init__(angle, axis)


class RotationZNode(RotationAxisAngleNode):

    def __init__(self, angle, dtype=np.float32):
        axis = np.array([0, 0, 1], dtype=dtype)
        super().__init__(angle, axis)


class GeometryNode(SceneNode):

    def __init__(self, geometry, visible=True):
        super().__init__(visible)
        self._geometry = geometry

    @property
    def geometry(self):
        return self._geometry

    def render(self, ctx, projection_mat, view_mat, model_mat):
        if self.visible:
            self._geometry.render(ctx, projection_mat, view_mat, model_mat)

    @property
    def children(self):
        return None

    @property
    def is_leaf(self):
        return True


class UniformVertices3DNode(GeometryNode):

    def __init__(self, vertices, render_mode, color, indices=None, dtype=np.float32):
        geom = geometry.UniformVertices3D(vertices, render_mode, color, indices, dtype)
        super().__init__(geom)


class UniformPoints3DNode(GeometryNode):

    def __init__(self, points, color, point_size=1.0, indices=None, dtype=np.float32):
        geom = geometry.UniformPoints3D(points, color, point_size, indices, dtype)
        super().__init__(geom)


class UniformLines3DNode(GeometryNode):

    def __init__(self, lines, colors, line_width=1.0, indices=None, dtype=np.float32):
        geom = geometry.UniformLines3D(lines, colors, line_width, indices, dtype)
        super().__init__(geom)


class UniformLineStrip3DNode(GeometryNode):

    def __init__(self, lines, color, line_width=1.0, indices=None, dtype=np.float32):
        geom = geometry.UniformLineStrip3D(lines, color, line_width, indices, dtype)
        super().__init__(geom)


class UniformTriangles3DNode(GeometryNode):

    def __init__(self, vertices, color, indices=None, dtype=np.float32):
        geom = geometry.UniformTriangles3D(vertices, color, indices, dtype)
        super().__init__(geom)


class Vertices3DNode(GeometryNode):

    def __init__(self, vertices, render_mode, color, indices=None, dtype=np.float32):
        geom = geometry.Vertices3D(vertices, color, render_mode, indices, dtype)
        super().__init__(geom)


class Points3DNode(GeometryNode):

    def __init__(self, points, colors, point_size=1.0, indices=None, dtype=np.float32):
        geom = geometry.Points3D(points, colors, point_size, indices, dtype)
        super().__init__(geom)


class Lines3DNode(GeometryNode):

    def __init__(self, lines, colors, line_width=1.0, indices=None, dtype=np.float32):
        geom = geometry.Lines3D(lines, colors, line_width,indices, dtype)
        super().__init__(geom)


class LineStrip3DNode(GeometryNode):

    def __init__(self, lines, colors, line_width=1.0, indices=None, dtype=np.float32):
        geom = geometry.LineStrip3D(lines, colors, line_width, indices, dtype)
        super().__init__(geom)


class Triangles3DNode(GeometryNode):

    def __init__(self, vertices, color, indices=None, dtype=np.float32):
        geom = geometry.Triangles3D(vertices, color, indices, dtype)
        super().__init__(geom)


class TextureTriangles3DNode(GeometryNode):

    def __init__(self, mesh_data, color=None):
        geom = geometry.TextureTriangles3D(mesh_data, color)
        super().__init__(geom)


class TextureMeshNode(GeometryNode):

    def __init__(self, mesh_data, color=None, visible=True):
        geom = geometry.TextureMesh(mesh_data, color)
        super().__init__(geom, visible)


class SceneGraph(object):

    def __init__(self):
        self._root = SceneNode()

    def render(self, ctx, projection_mat, view_mat, model_mat=None):
        if model_mat is None:
            model_mat = np.eye(4, dtype=view_mat.dtype)
        self._root.render(ctx, projection_mat, view_mat, model_mat)

    @property
    def root(self):
        return self._root

