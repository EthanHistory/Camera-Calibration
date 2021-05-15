import numpy as np
import cv2 as cv
assert cv.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0 but version is ' + str(cv.__version__[0])

# param = [K (3 x 3) : camera matrix, q (1 x 6): radial distortion coefficients, R_n (n x 3 x 3) : Rodrigues rotation vecotr of n camera,
#          t_n (n x 3 x 1) : translation vector of n camera, X_ni (n x i x 3) : 3D point in world coordinates]
def params_dict(n, i, K, q, R, t, X) -> np.array :
        return dict({"n": n, "i" : i, "K" : K, "q" : q, "R" : R, "t" : t, "X" : X})

class radial_distortion:
    def __init__(self):
        pass

    def __homogeneous_to_cartesian(self, X : np.array) -> np.array :
        assert(len(X.shape) == 2)
        return X[:, :-1] / X[:, [-1]]

    def __cartesian_to_homogeneous(self, X : np.array) -> np.array :
        assert(len(X.shape) == 2)
        Y = np.ones(list(X.shape[:-1]) + [1], float)

        return np.append(X, Y, axis=1)

    def __radius(self, X : np.array) -> np.array :
        assert(len(X.shape) == 2)
        assert(X.shape[1] == 2)
        return np.linalg.norm(X, axis=1)

    def __radial_distortion_function(self, r : np.array, dist_coff : np.array) -> np.array :
        assert(len(dist_coff.shape) == 1)
        assert(dist_coff.shape[0] == 6)
        return (1 + dist_coff[0] * (r ** 2) + dist_coff[1] * (r ** 4) + dist_coff[2] * (r ** 6)) / \
                (1 + dist_coff[3] * (r ** 2) + dist_coff[4] * (r ** 4) + dist_coff[5] * (r ** 6))

    def radial_distortion_model(self, param : dict) -> np.array :

        n = param['n']
        q = np.reshape(param['q'], [-1])
        K = param['K']
        i = param["i"]
        ud_points_set = np.empty((0, i, 2), float)
        for camera in range(n):
            R = param['R'][camera]
            t = param['t'][camera]
            R_matrix, _ = cv.Rodrigues(R)
            R_T = np.c_[R_matrix, t] # 3 x 4
            
            X = param["X"][camera]
            X_zero_one = np.c_[X, np.ones([i, 1], dtype=float)] # (40 x 4)
            
            camera_points = np.matmul(R_T, X_zero_one.transpose())
            # pixel_points = np.matmul(K, camera_points)
            # distorted_points = homogeneous_to_cartesian(pixel_points.transpose())

            # r = radius(distorted_points)
            # nth_image_ud_points = distorted_points * np.reshape(radial_distortion_function(r, q), [i, 1])
            # ud_points = np.append(ud_points, nth_image_ud_points.reshape(1, i, 2), axis=0)
            distorted_points = self.__homogeneous_to_cartesian(camera_points.transpose())

            r = self.__radius(distorted_points)
            ud_points = distorted_points * np.reshape(self.__radial_distortion_function(r, q), [i, 1])

            homo_ud_points = self.__cartesian_to_homogeneous(ud_points)
            pixel_ud_points = np.matmul(K, homo_ud_points.transpose())
            ud_points = self.__homogeneous_to_cartesian(pixel_ud_points.transpose())

            ud_points_set = np.append(ud_points_set, ud_points.reshape(1, i, 2), axis=0)
        return ud_points_set

    # def residual(x : np.array, param : dict) -> float:
    #     return np.abs(x - radial_distortion_model(param))

    # def jacobian_u_over_w(x : np.array, param : list) -> np.array :

    #     pm = opencv_polynomial_model(x, k)
    #     em = equidistant_model(x, focal)

    #     row = [[2 * (p - e) * (x_ ** 3), 
    #     2 * (p - e) * (x_ ** 5)] for p, e, x_ in zip(pm, em, x)]

    #     return np.array(row)

    # def jacobian_p_over_residual(param : list) -> np.array :

    #     pm = opencv_polynomial_model(x, k)
    #     em = equidistant_model(x, focal)

    #     row = [[2 * (p - e) * (x_ ** 3), 
    #     2 * (p - e) * (x_ ** 5)] for p, e, x_ in zip(pm, em, x)]

    #     return np.array(row)

    # def jacobian(x : np.array, param : list) -> np.array :

    #     row = np.empty((0, ), float)
    #     rdm = radial_distortion_model(param)
    #     row = 2 (rdm - x) * jacobian_p_over_residual(param)
    #     2 (rdm - x) * jacobian_p_over_residual(param)
    #     row = [ for p, e, x_ in zip(pm, em, x)]

    #     return np.array(row)
