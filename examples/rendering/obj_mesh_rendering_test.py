import numpy as np
import moderngl
from pybh import transforms
from pybh import math_utils
from pybh.rendering import geometry
from pybh.rendering import scene
from pybh.rendering import pyassimp_mesh
from pybh.rendering import pyglet_utils
from pybh.rendering import opengl_camera
# from pybh import pymesh_mesh
# from OpenGL import GL as gl


viewer = pyglet_utils.PygletSceneViewer(vsync=True)
# viewer = pyglet_utils.PygletViewer(vsync=True)
# camera = opengl_camera.Camera(window.width, window.height)
# window.register_camera(camera)
# camera = opengl_camera.TrackballCamera(viewer.width, viewer.height)
# viewer.set_trackball(camera)
# ctx = moderngl.create_context(330)

# ctx = moderngl.create_standalone_context(330)
ctx = viewer.ctx

axis = geometry.Axis3D(scale=8.0, line_width=5.0)

obj_mesh = pyassimp_mesh.PyassimpMesh("meshes/Rabbit.obj")
# obj_mesh = pyassimp_mesh.PyassimpMesh("meshes/earth.obj")
# obj_mesh = pyassimp_mesh.PyassimpMesh("meshes/uv_sphere.obj")
# obj_mesh = pyassimp_mesh.PyassimpMesh("meshes/capsule.obj")
light = geometry.PhongLight(position=[0, 10, 10], Ka=0.5, Kd=0.5, Ks=0.3, attenuate=False, falloff_factor=0.02)
mesh_node = scene.TextureMeshNode(obj_mesh.mesh_data, light=light)
# mesh_node.geometry.set_shader_mode(geometry.TextureTriangles3D.SHADER_MODE_NORMAL)
# mesh_node.geometry.set_render_phong_shading(False)
for td in obj_mesh.mesh_data:
    bbox = math_utils.BoundingBox.from_points(td.vertices)
    print("Bbox:", bbox, " volume:", bbox.volume())

graph = scene.SceneGraph()

rotation_node_x = scene.RotationXNode(0*1. * np.pi / 2)
rotation_node_y = scene.RotationYNode(-0 * 0.5 * np.pi / 4)
scale_node = scene.ScaleNode(1.0)
# scale_node = scene.ScaleNode(5.0)
scale_node.add_child(mesh_node)
scale_node.add_child(axis)
rotation_node_x.add_child(scale_node)
rotation_node_y.add_child(rotation_node_x)
graph.root.add_child(rotation_node_y)

# width = viewer.width
# height = viewer.height
# fbo = ctx.simple_framebuffer((width, height))
# fbo.use()

ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.CULL_FACE)
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

viewer.add_scene(graph)

viewer.trackball.look_at([0, 0, 15], [0, 0, 0], [0, 1, 0])

#ctx.wireframe = True
# mesh_node.geometry.render_normal = True
# mesh_node.geometry.raw_normal = True
# mesh_node.geometry.render_texture = False


while not viewer.should_quit:
    viewer.render()

    mat = transforms.TransformMatrix.from_axis_rotation(0.5 * np.pi / 180, [0, 1, 0]).matrix
    rotation_node_x.transformation = np.dot(mat, rotation_node_x.transformation)
    # # mat = transforms.TransformMatrix.from_axis_rotation(0.5 * np.pi / 180, [1, 0, 0]).matrix
    # # rotation_node_x.transformation = np.dot(mat, rotation_node_x.transformation)
