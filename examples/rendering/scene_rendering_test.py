import numpy as np
from PIL import Image
import moderngl
import geometry
import scene
from opengl_math import perspective_matrix


ctx = moderngl.create_standalone_context(330)

vertices = []
colors = []
texcoords = []
z = 0
# for i in range(0, 65):
#     # grid += struct.pack('6f', i - 32, -32.0, z, i - 32, 32.0, z)
#     # grid += struct.pack('6f', -32.0, i - 32, z, 32.0, i - 32, z)
#     vertices.append(np.array([i - 32, -32.0, z]))
#     vertices.append(np.array([i - 32, 32.0, z]))
#     vertices.append(np.array([-32.0, i - 32, z]))
#     vertices.append(np.array([32.0, i - 32, z]))
vertices.append(np.array([-10, -10, z]))
colors.append(np.array([1, 0, 0, 1]))
texcoords.append(np.array([0, 0]))
vertices.append(np.array([+10, -10, z]))
colors.append(np.array([1, 1, 0, 1]))
texcoords.append(np.array([1, 0]))
vertices.append(np.array([+10, +10, z]))
colors.append(np.array([0, 1, 0, 1]))
texcoords.append(np.array([1, 1]))
vertices.append(np.array([-10, +10, z]))
colors.append(np.array([0, 1, 1, 1]))
texcoords.append(np.array([0, 1]))
vertices.append(np.array([-10, -10, z]))
colors.append(np.array([0, 0, 1, 1]))
texcoords.append(np.array([0, 0]))

lines3d = scene.UniformLineStrip3DNode(vertices, np.array([0, 1, 0, 1], dtype=np.float32))
lines3d = scene.LineStrip3DNode(vertices, colors)
# lines3d = UniformLines3D(vertices, np.array([0, 1, 0, 1], dtype=np.float32))
# points3d = geometry.UniformPoints3D(vertices, np.array([0, 0, 1, 1], dtype=np.float32))
points3d = scene.Points3DNode(vertices, colors)
points3d.geometry.set_point_size(10.0)

indices = [0, 1, 2, 2, 3, 0]
faces = [[0, 1, 2], [2, 3, 0]]
# triangles3d = scene.Triangles3DNode(vertices, colors, indices)
# triangles3d = scene.Triangles3DNode(vertices, colors, faces)
# triangles3d = scene.Triangles3DNode(vertices[:3], colors[:3])

# texture = 128 * np.ones((50, 50, 3), dtype=np.uint8)
texture = geometry.LazyTexture.from_file("texture3.jpg")
triangles_data = geometry.TextureTrianglesData(vertices, texcoords, texture, indices=faces)
triangles3d = scene.TextureTriangles3DNode(triangles_data)
# triangles3d.geometry.use_texture = False
triangles3d.geometry.color = [0.0, 0.5, 0.5, 1]

mesh_data = geometry.TextureTrianglesData(vertices, texcoords, texture, indices=faces)
mesh_node = scene.TextureMeshNode(mesh_data)

graph = scene.SceneGraph()
# graph.root.add_child(scene.GeometryNode(lines3d))
# graph.root.add_child(scene.GeometryNode(points3d))
rotation_node = scene.RotationYNode(np.pi / 4)
translation_node = scene.TranslationNode([4, 0, 0])
scale_node = scene.ScaleNode(0.5)
scale_node.add_child(lines3d)
scale_node.add_child(points3d)
# scale_node.add_child(triangles3d)
scale_node.add_child(mesh_node)
translation_node.add_child(scale_node)
rotation_node.add_child(translation_node)
graph.root.add_child(rotation_node)

width = 512
height = 512
fbo = ctx.simple_framebuffer((width, height))
fbo.use()

ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.CULL_FACE)
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

projection_mat = perspective_matrix(90 * np.pi / 180, width / height, 0.1, 1000.)
view_mat = np.eye(4, dtype=np.float32)
translation = np.array([0, 0, -15], dtype=np.float32)
# translation = np.array([0, 0, 5], dtype=np.float32)
view_mat[:3, 3] = translation

fbo.clear(0.3, 0.3, 0.3, 1.0)
graph.render(ctx, projection_mat, view_mat)
