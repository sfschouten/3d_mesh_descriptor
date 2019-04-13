import math
import itertools
import numpy as np
import trimesh
from numpy.linalg import norm
from trimesh.visual import create_visual
from scipy.sparse import lil_matrix
from trimesh.ray.ray_triangle import ray_triangle_id
from trimesh.triangles import barycentric_to_points, points_to_barycentric
from abc import ABC, abstractmethod


class SphericalParameterization(ABC):
    
    class NotBuiltException(Exception):
        pass

    @abstractmethod
    def sample(self, nlat, nlon):
        pass

    @abstractmethod
    def build():
        pass

class FastSphericalParameterization(SphericalParameterization):
    
    def __init__(self, mesh, alpha = 1):
        self.mesh = mesh
        self.A = self._calculate_A(alpha)

    def _custom_intersect(self, mesh, origins, directions):
        return ray_triangle_id( triangles        = mesh.triangles,
                                ray_origins      = origins,
                                ray_directions   = directions,
                                tree             = mesh.triangles_tree,
                                multiple_hits    = False,
                                triangles_normal = mesh.face_normals)


    def sample(self, lat, lon):
        if not self.built:
            raise NotBuiltException()

        """
        1. Cast rays from origin to samplepoints
        2. Find points at which rays intersect our unit sphere mesh
        3. Convert points to barycentric coordinates
        4. Find corresponding points in original mesh
        5. Return x,y,z values found for each sample point
        """
        
        values  = [] 
        for v in range(lat):
            start_points = np.zeros((lon, 3))
            directions = np.zeros((lon, 3))
            
            phi = (math.pi * v) / lat 
            for u in range(lon):
                theta = (2 * math.pi * u) / lon
                
                directions[u,0] = math.sin(phi) * math.cos(theta)
                directions[u,1] = math.sin(phi) * math.sin(theta)
                directions[u,2] = math.cos(phi)                  

            M = self.mesh
            Q = self.Q           
            
            (index_tri, 
            index_ray, 
            locations) = self._custom_intersect(Q, start_points, directions)
            
            sphere_triangles   = np.array( [[Q.vertices[w] for w in Q.faces[i]] for i in index_tri]  )
            original_triangles = np.array( [[M.vertices[w] for w in M.faces[i]] for i in index_tri]  )
            
            bary_points = points_to_barycentric(sphere_triangles, locations)
            values.append(barycentric_to_points(original_triangles, bary_points))
        
        return np.array(values)

    def _calculate_A(self, alpha):
        mesh = self.mesh

        N = len(mesh.vertices)
        result = lil_matrix((N,N))
    
        for i,v in enumerate(mesh.vertices):
            n_i = len(mesh.vertex_neighbors[i])
            result[i,i] = ((n_i - 1) * n_i + alpha * n_i + n_i * 4 / 3) / (n_i * (n_i + 5))

        for k,(i,j) in enumerate(mesh.edges):
            n_i = len(mesh.vertex_neighbors[i])
            result[i,j] = (2 - alpha + 4/3 + 4/3) / (n_i * (n_i + 5))

        return result.tocsc()


    def _project2unitsphere(self, mesh):
        for i,v in enumerate(mesh.vertices):
            mesh.vertices[i] = v / np.linalg.norm(v)


    def _vertex2face(self, mesh):
        result = { i : set() for i,__ in enumerate(mesh.vertices) }

        for k,(i,j) in enumerate(mesh.edges):
            f = mesh.edges_face[k]
            result[i].add(f)
            result[j].add(f)

        return result


    def build(self, s, n):
        M = self.mesh.copy()
        A = self.A

        # Smooth the mesh for better center
        verts = M.vertices
        for i in range(s):
            verts = A.dot(verts)

        M.vertices = verts

        # Center the mesh
        C = M.centroid
        M.apply_translation(-C)

        # Project to unit circle        
        self._project2unitsphere(M)

        # Smooth to remove overlap
        Q_i = M.copy()

        faceLookup = self._vertex2face(Q_i)
        
        u = 0.25
        print("Vertex Count: ", len(Q_i.vertices))
        for i in range(n):
            Q_i_prev_vertices = Q_i.vertices.copy()
            print(i)

            # Smooth
            Q_i.vertices = A.dot(Q_i.vertices)

            # Add difference u times more
            diff         = np.subtract(Q_i.vertices, Q_i_prev_vertices)
            Q_i.vertices = np.add(Q_i.vertices, u * diff)

            # Put vertices in center of neighbours
            faceCenters = Q_i.triangles_center
            faceAreas = Q_i.area_faces
            faceAngles = Q_i.face_angles
            newVerts = Q_i.vertices.copy()
            sphericalTriangleAreas = faceAreas.copy()
             
            vectorangleF = lambda vs: math.acos( np.dot(*vs) / (norm(vs[0]) * norm(vs[1])))

            for f in range(len(Q_i.faces)):
                vertices = Q_i.vertices[Q_i.faces[f]]
                sphericalTriangleAreas[f] = sum( map(vectorangleF, itertools.combinations(vertices, 2) ) )
                    
            for j in range(len(Q_i.vertices)):
                newV = np.array([0,0,0])
                totalArea = 0
                for f in faceLookup[j]:
                    totalArea += sphericalTriangleAreas[f]
                    centroid = faceCenters[f] / np.linalg.norm(faceCenters[f])
                    newV = np.add(newV, sphericalTriangleAreas[f] * centroid)
                
                newV /= totalArea
                newVerts[j] = newV

            Q_i.vertices = newVerts
#            if u > 0.25:
#                u -= 10 / (n - 100)

            # Project back to unit circle
            self._project2unitsphere(Q_i)


        self.built = True
        self.Q = Q_i
