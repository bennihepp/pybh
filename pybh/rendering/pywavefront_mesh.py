import os
import numpy as np
from geometry import TextureTrianglesData
from moderngl_utils import LazyTexture
import pywavefront


class WavefrontObjMesh(object):

    def __init__(self, filename, dtype=np.float32, int_dtype=np.int32):
        if not os.path.isfile(filename):
            raise IOError("No such file: {:s}".format(filename))
        meshes = pywavefront.Wavefront(filename)
        self._mesh_data = []
        for material_name, material in meshes.materials.items():
            data = np.asarray(material.vertices, dtype=dtype)
            data = data.reshape(-1, 8)
            texcoords = np.ascontiguousarray(data[:, :2])
            normals = np.ascontiguousarray(data[:, 2:5])
            vertices = np.ascontiguousarray(data[:, 5:])
            image_name = material.texture.image_name
            image_filename = os.path.join(os.path.dirname(filename), image_name)
            texture = LazyTexture.from_file(image_filename)
            triangles_data = TextureTrianglesData(vertices, texcoords, texture, normals, dtype=dtype, int_dtype=int_dtype)
            self._mesh_data.append(triangles_data)

    @property
    def mesh_data(self):
        return self._mesh_data
