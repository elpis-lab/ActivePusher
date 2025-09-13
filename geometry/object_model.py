import numpy as np
import trimesh


def get_obj_shape(obj_name):
    """Get the shape of the object"""
    mesh = trimesh.load(obj_name)
    shape = mesh.bounds[1] - mesh.bounds[0]
    return shape
