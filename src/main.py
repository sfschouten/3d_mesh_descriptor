import numpy as np
import trimesh
import spherical_parameterization as sp
import pyshtools 
from pyshtools.shclasses.shcoeffsgrid import SHGrid

#mesh = trimesh.load("../suzanne.obj")
mesh = trimesh.load("../fuze.obj")
#mesh = trimesh.load("../cube.obj")
#mesh = trimesh.load("../armadillo.obj")
#mesh = trimesh.load("../hand.obj")


p = sp.FastSphericalParameterization(mesh)
p.build(100, 200)

#p.Q.show()
#exit()

U = 200; V = 400

samples = p.sample(U, V)
x_grid = SHGrid.from_array(samples[:,:,0])
y_grid = SHGrid.from_array(samples[:,:,1])
z_grid = SHGrid.from_array(samples[:,:,2])

x_coeffs = x_grid.expand()
y_coeffs = y_grid.expand()
z_coeffs = z_grid.expand()

x_coords = x_coeffs.expand(grid='DH2').to_array()
y_coords = y_coeffs.expand(grid='DH2').to_array()
z_coords = z_coeffs.expand(grid='DH2').to_array()

coords = [list(zip(xs, ys, zs)) for xs, ys, zs in zip(x_coords, y_coords, z_coords)]

vertex_list = []
vertices = {}
v = 0
for col in coords:
    for coord in col:
        if coord not in vertices:
            vertices[coord] = v
            vertex_list.append(coord)
            v += 1
        

faces = []
for u in range(-1, U-1):
    for v in range(-1, V-1):
        i = vertices[coords[u][v]]
        j = vertices[coords[u+1][v]]
        k = vertices[coords[u][v+1]]
        l = vertices[coords[u+1][v+1]]
        faces.append((i,j,l))
        faces.append((i,l,k))
        
new_mesh = trimesh.Trimesh(vertex_list, faces)
new_mesh.show()

"""

TODO:
    1. simply convert back to a mesh.
        * create vertex for every unique location
        * create faces between vertices according to locations grid.

    2. try and truncate part of the coefficients to simplify the mesh.
    3. linear interpolation between 2 meshes.

"""
