import os
import numpy as np
import pyassimp
from .geometry import TextureTrianglesData
from .moderngl_utils import LazyTexture


class PyassimpMesh(object):

    def __init__(self, filename, dtype=np.float32, int_dtype=np.int32, compute_normals_if_invalid=True, verbose=True):
        if not os.path.isfile(filename):
            raise IOError("No such file: {:s}".format(filename))
        scene_obj = pyassimp.load(filename)
        if verbose:
            print("Loaded scene from file {:s}".format(filename))
        assert len(scene_obj.meshes) > 0
        self._mesh_data = []
        for mesh in scene_obj.meshes:
            vertices = np.ascontiguousarray(mesh.vertices, dtype=dtype).reshape(-1, 3)
            if len(mesh.normals) > 0:
                normals = np.ascontiguousarray(mesh.normals, dtype=dtype).reshape(-1, 3)
            else:
                normals = None
            faces = np.ascontiguousarray(mesh.faces, dtype=int_dtype).reshape(-1, 3)
            texcoords = None
            texture = None
            colors = None
            if len(mesh.colors) > 0:
                colors = np.ascontiguousarray(mesh.colors, dtype=dtype).reshape(-1, 4)
            if len(mesh.texturecoords) > 0:
                assert mesh.texturecoords.ndim == 3
                assert mesh.texturecoords.shape[2] == 3
                texcoords = np.ascontiguousarray(mesh.texturecoords[0, :, :2], dtype=dtype).reshape(-1, 2)
                material = mesh.material
                # Workaround for modified pyassimp properties dict
                properties = {key: value for key, value in material.properties.items()}
                if 'file' in properties:
                    image_filename = properties['file']
                    image_filename = os.path.join(os.path.dirname(filename), image_filename)
                    texture = LazyTexture.from_file(image_filename)
            td = TextureTrianglesData(vertices, texcoords=texcoords, texture=texture, colors=colors, normals=normals, indices=faces,
                dtype=dtype, int_dtype=int_dtype, compute_normals_if_invalid=compute_normals_if_invalid)
            self.mesh_data.append(td)
        pyassimp.release(scene_obj)

    @property
    def mesh_data(self):
        return self._mesh_data
