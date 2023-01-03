import math
import numpy as np
import cv2 as cv

def normalize(points):
    '''
    Normalize coordinates of points (centroid to the origin and mean distance of sqrt(2 or 3))
    
    Input
    -----
    
    Output
    -----
    
    '''
    
    # points = np.asfarray(points, dtype=np.float64)
    mean_, std_ = np.mean(points, axis=0), np.std(points, axis=0)
    T = [[std_[0], 0, mean_[0]], [0, std_[1], mean_[1]], [0, 0, 1]]
    T = np.linalg.inv(T)
    homogeneous_points = np.concatenate((points.T, np.ones([1, points.shape[0]])))
    normalized_points = np.dot(T, homogeneous_points).T[:, :2]
    return normalized_points, T

def find_homography(objpoints, imgpoints):
        '''
        Solve the equation of x, x ~ Hy, by transfroming it into Ax = 0 
        Input
        -----
        A: number of dimensions, 3 here
        x: the data to be normalized (directions at different columns and points at rows)
        Output
        ------
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
        '''
        assert(len(objpoints) == len(imgpoints))
        
        num_points = len(objpoints)
        
        normalized_objpoints, objT = normalize(objpoints)
        normalized_imgpoints, imgT = normalize(imgpoints)
        
        A = np.empty((0, 9), float)
        for i in range(num_points):
            x, y = normalized_objpoints[i]
            u, v = normalized_imgpoints[i]
            A = np.append(A, np.array([[-x, -y, -1, 0, 0, 0, u * x, u * y, u]]), axis=0)
            A = np.append(A, np.array([[0, 0, 0, -x, -y, -1, v * x, v * y, v]]), axis=0)
        U, S, V = np.linalg.svd(A, full_matrices=False)
        
        H = np.dot(np.linalg.inv(imgT), np.dot(np.array(V[-1]).reshape([3,3]), objT))
        H = H / H[2][2]
        return H