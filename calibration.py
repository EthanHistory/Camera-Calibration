from typing import List

import math
import numpy as np
import cv2 as cv

from sympy.matrices import Matrix, eye
from sympy import symbols, sqrt, sin, cos, lambdify

from tqdm import tqdm
import scipy.linalg as la

def __normalize(points):
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

def find_homography(objpoints:np.array, imgpoints:np.array):
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
        
        normalized_objpoints, objT = __normalize(objpoints)
        normalized_imgpoints, imgT = __normalize(imgpoints)
        
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
    
def __DLT_for_B(Hs:np.array) -> np.ndarray:
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
            
        # skewless constraint
        A = np.append(A, np.array([0, 1, 0, 0, 0, 0]).reshape([1, -1]), axis=0)

        U, S, V = np.linalg.svd(A, full_matrices=False)
        B = V[-1]
        return B

def __init_camera_matrix(Hs:np.array) -> np.array :
    B = __DLT_for_B(Hs)

    # result of Cholesky decomposition
    v0 = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1] * B[1])
    lambda_ = B[5] - (B[3] * B[3] + v0 * (B[1] * B[3] - B[0] * B[4])) / B[0]
    alpha_ = math.sqrt(lambda_ / B[0])
    beta_ = math.sqrt(lambda_ * B[0] / (B[0] * B[2] - B[1] * B[1]))
    gamma_ = -B[1] * alpha_ * alpha_ * beta_ / lambda_
    u0 = gamma_ * v0 / beta_ - B[3] * alpha_ * alpha_ / lambda_
    
    # skewless
    gamma_ = 0
    
    K = np.array([alpha_, gamma_, u0, 0, beta_, v0, 0, 0, 1]).reshape(3, 3)

    return K

def __init_camera_extrinsics(Hs:np.array) -> np.array :
    K = __init_camera_matrix(Hs)
    
    K_inv = np.linalg.inv(K)

    rvecs = np.empty((0, 3, 1), float)
    tvecs = np.empty((0, 3, 1), float)

    for H in Hs:
        r1 = np.matmul(K_inv, H[:, 0]).reshape(3,1)
        r1_norm = np.linalg.norm(r1, ord=2)
        r1 = r1 / r1_norm

        r2 = np.matmul(K_inv, H[:, 1]).reshape(3,1)
        r2_norm = np.linalg.norm(r2, ord=2)
        r2 = r2 / r2_norm

        r3 = np.cross(r1, r2, axis=0)
        
        t = np.matmul(K_inv, H[:, 2]).reshape(3,1)
        tvec = t / r1_norm

        Q = np.concatenate((r1, r2, r3), axis=1)
        U, S, V = np.linalg.svd(Q, full_matrices=False)
        R = np.matmul(U, V.transpose())

        rvec, _ = np.array(cv.Rodrigues(R), dtype='object')
        rvecs = np.append(rvecs, rvec.reshape(1, 3, 1), axis=0)
        tvecs = np.append(tvecs, tvec.reshape(1, 3, 1), axis=0)

    return rvecs, tvecs

def parameter_initialization(objpoints:np.array, imgpoints:np.array):
    assert(objpoints.shape[0] == imgpoints.shape[0])
    assert(objpoints.shape[2] == 3)
    assert(imgpoints.shape[2] == 2)
    
    nviews = objpoints.shape[0]
    Hs = []
    for i in range(nviews):
        Hs.append(find_homography(objpoints[i][:, :2], imgpoints[i]))
        
    K = __init_camera_matrix(Hs)
    rvecs, tvecs = __init_camera_extrinsics(Hs)
    dist = np.array([0, 0, 0, 0, 0])
    
    return K, dist, rvecs, tvecs

def __transform_parameters(param_1d, n):
    dist = np.array([param_1d[0], param_1d[1], 0, 0, 0]).reshape(5, 1)
    mtx = np.array([[param_1d[2], 0, param_1d[4]], [0, param_1d[3], param_1d[5]], [0, 0, 1]])
    rvecs = param_1d[6:6+3*n].reshape(-1, 3)
    tvecs = param_1d[6+3*n:].reshape(-1, 3)
    return mtx, dist, rvecs, tvecs

def __compute(objpoints, imgpoints, param_1d, J_func, res_func):
    N = len(objpoints)
    M = len(objpoints[0])
    jacobian = np.zeros(shape=[N*M, 6+6*N])
    residual = np.zeros(shape=[N*M, 1])
    for n in range(N):
            for i in range(M):
                # Set X_world and p
                values = {"px": imgpoints[n][i][0], "py": imgpoints[n][i][1],
                        "x": objpoints[n][i][0], "y": objpoints[n][i][1], "z": objpoints[n][i][2],
                        "q1": param_1d[0], "q2": param_1d[1],
                        "alpha": param_1d[2], "beta": param_1d[3], "u0": param_1d[4], "v0": param_1d[5],
                        "kx": param_1d[6+3*n], "ky": param_1d[7+3*n], "kz": param_1d[8+3*n],
                        "tx": param_1d[6+3*N+3*n], "ty": param_1d[7+3*N+3*n], "tz": param_1d[8+3*N+3*n]}
                jac = J_func(**values).flatten()
                # General parameter
                jacobian[i+M*n, :6] = jac[:6]
                # rotation vector
                jacobian[i+M*n, 6+3*n:9+3*n] = jac[6:9]
                # translation vector
                jacobian[i+M*n, 6+3*N+3*n:9+3*N+3*n] = jac[9:]
                residual[i+M*n] = res_func(**values).flatten()
    return jacobian, residual

def calibration(objpoints:np.array, imgpoints:np.array, niter=20):
    # Set up jacobian and residual

    print("Defining sympy symbols")
    # inputs
    x, y, z = symbols("x y z")
    px, py = symbols('px py')
    inputs = [x, y, z, px, py]
    
    # params
    kx, ky, kz = symbols("kx ky kz")
    tx, ty, tz = symbols("tx ty tz")
    q1, q2 = symbols(f'q1 q2')
    alpha, beta, u0, v0 = symbols(f'alpha beta u0 v0') 
    params = [q1, q2, alpha, beta, u0, v0, kx, ky, kz, tx, ty, tz]
    
    print("Calculating jacobian and residual of objective over params")
    X_world = Matrix(3, 1, [x, y, z])
    theta = sqrt(kx**2 + ky**2 + kz**2)
    kr = Matrix(3, 1, [kx, ky, kz]) / theta
    K = Matrix(3, 3, [0, -kr[2], kr[1],
                    kr[2], 0, -kr[0],
                    -kr[1], kr[0], 0])
    T = Matrix(3, 1, [tx, ty, tz])
    
    X_camera = (cos(theta)*eye(3) + (1-cos(theta))*kr*kr.T + sin(theta)*K)*X_world + T
    x_norm = Matrix(2, 1, [X_camera[0] / X_camera[2], X_camera[1] / X_camera[2]])
    
    r = x_norm.T * x_norm
    f = q1 * r + q2 * r**2
    
    x_dist = x_norm + x_norm * f
    u_dist = Matrix(2, 1, [alpha*x_dist[0] + u0, beta*x_dist[1] + v0])
    
    p = Matrix(2, 1, [px, py])
    diff = p-u_dist
    res = diff.T * diff
    
    J = res.jacobian(params)
    J_func = lambdify(params+inputs, J, modules='numpy')
    res_func = lambdify(params+inputs, res, modules='numpy')

    print("Optimizing params")
    N = len(objpoints)
    M = len(objpoints[0])
    
    mtx, dist, rvecs, tvecs = parameter_initialization(objpoints, imgpoints)
    param_1d = np.array([dist[0], dist[1], mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]] + list(rvecs.flatten()) + list(tvecs.flatten()))
    param_history = [__transform_parameters(param_1d, N)]
    
    for iter in tqdm(range(niter)):
        jacobian, residual = __compute(objpoints, imgpoints, param_1d, J_func, res_func)
        update = la.lstsq(jacobian, residual)[0]
        param_1d -= update.flatten()
        param_history.append(__transform_parameters(param_1d, N))
        print(f"[iter {iter}] residual: {np.sum(np.sqrt(residual))})")
    
    _, residual = __compute(objpoints, imgpoints, param_1d, J_func, res_func)
    print(f"Final residual: {np.sum(np.sqrt(residual))})")
    
    mtx, dist, rvecs, tvecs = __transform_parameters(param_1d, N)
    return mtx, dist, rvecs, tvecs, param_history