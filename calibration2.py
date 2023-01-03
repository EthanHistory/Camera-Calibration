from typing import List

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
    
def DLT_for_B(Hs:list) -> np.ndarray:
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
        A = np.empty((0, 6), float)

        for H in Hs:
            assert(type(H) == np.ndarray)
            C_index_pair = [(0,1), (0,0), (1,1)]
            C = [[H[0][i] * H[0][j], H[0][i] * H[1][j] + H[1][i] * H[0][j], H[1][i] * H[1][j],
                H[2][i] * H[0][j] + H[0][i] * H[2][j], H[2][i] * H[1][j] + H[1][i] * H[2][j], H[2][i] * H[2][j]] for (i, j) in C_index_pair]
        
            first_row = np.array(C[0]).reshape([1, -1])
            second_row = (np.array(C[1]) - np.array(C[2])).reshape([1, -1])
            A = np.append(A, first_row, axis=0)
            A = np.append(A, second_row, axis=0)

        U, S, V = np.linalg.svd(A, full_matrices=False)
        B = V[-1]
        return B
    
def n_view_homographies(objpoints:List[np.array], imgpoints:List[np.array]):
    nviews = len(objpoints)
    Hs = []
    for i in range(nviews):
        Hs.append(find_homography(objpoints[i][:, :2], imgpoints[i]))
    return Hs

def init_camera_matrix(objpoints:List[np.array], imgpoints:List[np.array]) -> np.array :
    Hs = n_view_homographies(objpoints, imgpoints)  
    B = DLT_for_B(Hs)

    # result of Cholesky decomposition
    v0 = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1] * B[1])
    lambda_ = B[5] - (B[3] * B[3] + v0 * (B[1] * B[3] - B[0] * B[4])) / B[0]
    alpha_ = math.sqrt(lambda_ / B[0])
    beta_ = math.sqrt(lambda_ * B[0] / (B[0] * B[2] - B[1] * B[1]))
    gamma_ = -B[1] * alpha_ * alpha_ * beta_ / lambda_
    u0 = gamma_ * v0 / beta_ - B[3] * alpha_ * alpha_ / lambda_

    # print(f'v0 = {v0}')
    # print(f'lambda_ = {lambda_}')
    # print(f'alpha_ = {alpha_}')
    # print(f'beta_ = {beta_}')
    # print(f'gamma_ = {gamma_}')
    K = np.array([alpha_, gamma_, u0, 0, beta_, v0, 0, 0, 1]).reshape(3, 3)

    return K

def init_camera_extrinsics(objpoints:List[np.array], imgpoints:List[np.array]) -> np.array :
    Hs = n_view_homographies(objpoints, imgpoints)
    K = init_camera_matrix(objpoints, imgpoints)
    
    K_inv = np.linalg.inv(K)

    rvecs = np.empty((0, 3, 1), float)
    tvecs = np.empty((0, 3, 1), float)

    for H in Hs:
        r1 = np.matmul(K_inv, H[0, :].reshape(3, 1))
        r1_norm = np.linalg.norm(r1, ord=1)
        r1 = r1 / r1_norm

        r2 = np.matmul(K_inv, H[1, :].reshape(3, 1))
        r2_norm = np.linalg.norm(r2, ord=1)
        r2 = r2 / r2_norm

        r3 = np.cross(r1, r2, axis=0)
        
        t = np.matmul(K_inv, H[2, :].reshape(3, 1))
        t_norm = np.linalg.norm(t, ord=1)
        tvec = t / t_norm

        Q = np.concatenate((r1, r2, r3), axis=1)
        U, S, V = np.linalg.svd(Q, full_matrices=False)
        R = np.matmul(U, V.transpose())

        rvec, _ = np.array(cv.Rodrigues(R), dtype='object')
        rvecs = np.append(rvecs, rvec.reshape(1, 3, 1), axis=0)
        tvecs = np.append(tvecs, tvec.reshape(1, 3, 1), axis=0)

    return rvecs, tvecs