import numpy as np
import cv2 as cv
assert cv.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0 but version is ' + str(cv.__version__[0])

# param = [K (3 x 3) : camera matrix, q (1 x 6): radial distortion coefficients, R_n (n x 3 x 1) : Rodrigues rotation vecotr of n camera,
#          t_n (n x 3 x 1) : translation vector of n camera, X_ni (n x i x 3) : 3D point in world coordinates]

def params_dict(n, i, K, q, R, t, X) -> np.array :
        return dict({"n": n, "i" : i, "K" : K, "q" : q, "R" : R, "t" : t, "X" : X})

class radial_distortion:
    def __init__(self):
        pass
    
    def __param_size(self, image_size, points_size):
        return 3 * 3 + 6 + image_size * 3 * 2 + image_size * points_size * 3 

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
        
        distorted_points_set = np.empty((0, i, 2), float)
        
        for camera in range(n):
            R = param['R'][camera]
            t = param['t'][camera]
            R_matrix, _ = cv.Rodrigues(R)
            R_T = np.c_[R_matrix, t] # 3 x 4
            
            X = param["X"][camera]
            X_zero_one = np.c_[X, np.ones([i, 1], dtype=float)] # (40 x 4)
            
            camera_points = np.matmul(R_T, X_zero_one.transpose())
            undistorted_points = self.__homogeneous_to_cartesian(camera_points.transpose())

            r = self.__radius(undistorted_points)
            
            distorted_points = undistorted_points * np.reshape(self.__radial_distortion_function(r, q), [i, 1])
            
            homo_distorted_points = self.__cartesian_to_homogeneous(distorted_points)
            pixel_distorted_points = np.matmul(K, homo_distorted_points.transpose())
            distorted_points = self.__homogeneous_to_cartesian(pixel_distorted_points.transpose())

            distorted_points_set = np.append(distorted_points_set, distorted_points.reshape(1, i, 2), axis=0)
        return distorted_points_set

    # def _residual(self, x : np.array, param : dict) -> float:
    #     return np.abs(x - self.radial_distortion_model(param))


    # def __jac_r_over_gr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_r_over_hr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_r(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_r_over_fr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_v(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_u(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w_over_v(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w_over_u(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    #     jac = {}
    #     jac['u'] = self.__jac_over_u(camera, param)
    #     jac['w'] = self.__jac_over_u(camera, param)

    #     return np.array(row)

    # def __jac_over_residual(self, camera : int, param : dict, ud_points : np.array, radius : np.array) -> np.array :
    #     num_points = param["i"]
    #     q = np.reshape(param['q'], [-1])
    #     rdf = self.__radial_distortion_function(radius, q).reshape(-1, 1)
        
    #     jac = {}
    #     jac['u/w'] = self.__jac_over_w_over_u(camera, param, ud_points) # 40
    #     jac['v/w'] = self.__jac_over_w_over_u(camera, param, ud_points)
    #     jac['f(r)'] = self.__jac_over_w_over_u(camera, param, ud_points)
        
    #     pixel_points = self.__homogeneous_to_cartesian(ud_points)
        
    #     jac_x = jac['u/w'] * rdf + pixel_points[:, 0] * jac['f(r)']
    #     jac_y = jac['v/w'] * rdf + pixel_points[:, 1] * jac['f(r)']
        
    #     # https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    #     result = np.empty((jac_x.size + jac_y.size,), dtype=jac_y.dtype)
    #     result[0::2] = jac_x
    #     result[1::2] = jac_y
    #     return result

    # def jacobian(self, x : np.array, param : dict) -> np.array :
        
    #     jacobian_callbacks = {
    #         'u/w': self.__jac_over_w_over_u,
    #         'v/w': self.__jac_over_w_over_v,
    #         'f(r)': self.__jac_over_w_over_v,
    #         'f(r)/r': self.__jac_r_over_fr,
    #         'h(r)/r': self.__jac_r_over_hr,
    #         'g(r)/r': self.__jac_r_over_gr,
    #         'r': self.__jac_over_r,
    #         'u': self.__jac_over_u,
    #         'v': self.__jac_over_v,
    #         'w': self.__jac_over_w
    #     }
    #     n = param['n']
    #     q = np.reshape(param['q'], [-1])
    #     K = param['K']
    #     i = param["i"]
        
    #     # 콜백을 지역변수로 하고 메모리로 사용
    #     # 파라미터를 각 카메라 별로 분리해주는 함수가 필요
    #     # 파라미터를 
        
    #     for camera in range(n):
    #         R = param['R'][camera]
    #         t = param['t'][camera]
    #         R_matrix, _ = cv.Rodrigues(R)
    #         R_T = np.c_[R_matrix, t] # 3 x 4
            
    #         X = param["X"][camera]
    #         X_zero_one = np.c_[X, np.ones([i, 1], dtype=float)] # (40 x 4)
            
    #         camera_points = np.matmul(R_T, X_zero_one.transpose())
    #         undistorted_points = self.__homogeneous_to_cartesian(camera_points.transpose())

    #         r = self.__radius(undistorted_points)
            
    #         distorted_points = undistorted_points * np.reshape(self.__radial_distortion_function(r, q), [i, 1])
            
    #         homo_distorted_points = self.__cartesian_to_homogeneous(distorted_points)
    #         pixel_distorted_points = np.matmul(K, homo_distorted_points.transpose())
    #         distorted_points = self.__homogeneous_to_cartesian(pixel_distorted_points.transpose())
            
    #         jacobian = 2 * (distorted_points - x[camera]) * self.__jac_over_residual(camera, param, camera_points, r) # 40 x 2 x num(p)
    #         jacobian_per_image.append(jacobian)

    #     jac = np.concatenate(jacobian_per_image, axis=0)
    #     return jac

class radial_distortion_test:
    def __init__(self):
        pass
    
    def __homogeneous_to_cartesian(self, X : np.array) -> np.array :
        if len(X.shape) == 2:
            return X[:, :-1] / X[:, [-1]]
        elif len(X.shape) == 3:
            return X[:, :, :-1] / X[:, :, [-1]]
            

    def __cartesian_to_homogeneous(self, X : np.array) -> np.array :
        Y = np.ones(list(X.shape[:-1]) + [1], float)

        return np.append(X, Y, axis=len(X.shape) - 1)

    def __radius(self, X : np.array) -> np.array :
        return np.linalg.norm(X, axis=(len(X.shape) - 1))

    def __radial_distortion_function(self, radius : np.array, dist_coff : np.array) -> np.array :
        assert(len(dist_coff.shape) == 1)
        assert(dist_coff.shape[0] == 6)
        return (1 + dist_coff[0] * (radius ** 2) + dist_coff[1] * (radius ** 4) + dist_coff[2] * (radius ** 6)) / \
                (1 + dist_coff[3] * (radius ** 2) + dist_coff[4] * (radius ** 4) + dist_coff[5] * (radius ** 6))

    def radial_distortion_model(self, param : dict) -> np.array :

        n = param['n']
        q = np.reshape(param['q'], [-1])
        K = param['K']
        i = param["i"]
        # param = [K (3 x 3) : camera matrix, q (1 x 6): radial distortion coefficients, R_n (n x 3 x 1) : Rodrigues rotation vecotr of n camera,
#          t_n (n x 3 x 1) : translation vector of n camera, X_ni (n x i x 3) : 3D point in world coordinates]
        # [R | t] X => (n x 3 x 3 + n x 3 x 1) * n x i x 3 + 1 (br) => (n x 3 x 4) * (4 x i x n) => n x i x 3
        
        full_R_T = np.empty((0, 3, 4), float) # n x 3 x 4
        full_X = self.__cartesian_to_homogeneous(param["X"]) # n x i x 3
        for camera in range(n):
            R = param['R'][camera]
            t = param['t'][camera]
            R_matrix, _ = cv.Rodrigues(R)
            R_T = np.c_[R_matrix, t] # 3 x 4
            full_R_T = np.append(full_R_T, R_T.reshape(1, 3, 4), axis=0)
            
        temp = np.transpose(full_R_T, axes=(0, 2, 1)) # n x 4 x 3
        full_undistorted_homo_points = np.matmul(full_X, temp)
        print(f'shape (full_undistorted_homo_points) : {full_undistorted_homo_points.shape}')
        full_undistorted_cart_points = self.__homogeneous_to_cartesian(full_undistorted_homo_points)
        print(f'shape (full_undistorted_cart_points) : {full_undistorted_cart_points.shape}')
        full_radius = self.__radius(full_undistorted_cart_points)
        print(f'shape (full_radius) : {full_radius.shape}')
        full_distorted_cart_points = full_undistorted_cart_points * np.reshape(self.__radial_distortion_function(full_radius, q), [n ,i, 1])
        print(f'shape (full_distorted_cart_points) : {full_distorted_cart_points.shape}')
        full_distorted_homo_points = self.__cartesian_to_homogeneous(full_distorted_cart_points)
        print(f'shape (full_distorted_homo_points) : {full_distorted_homo_points.shape}')
        
        full_K_T = np.repeat(K.transpose().reshape(1, 3, 3), n, axis=0)
        print(f'shape (full_K_T) : {full_K_T.shape}')
        
        full_distorted_pixel_points = np.matmul(full_distorted_homo_points, full_K_T)
        full_distorted_pixel_points = self.__homogeneous_to_cartesian(full_distorted_pixel_points)
        print(f'shape (full_distorted_pixel_points) : {full_distorted_pixel_points.shape}')
        
        return full_distorted_pixel_points

    # def _residual(self, x : np.array, param : dict) -> float:
    #     return np.abs(x - self.radial_distortion_model(param))


    # def __jac_r_over_gr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_r_over_hr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_r(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_r_over_fr(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_v(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_u(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w_over_v(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    # def __jac_over_w_over_u(self, camera : int, param : dict, ud_points : np.array) -> np.array :
    #     jac = {}
    #     jac['u'] = self.__jac_over_u(camera, param)
    #     jac['w'] = self.__jac_over_u(camera, param)

    #     return np.array(row)

    # def __jac_over_residual(self, camera : int, param : dict, ud_points : np.array, radius : np.array) -> np.array :
    #     num_points = param["i"]
    #     q = np.reshape(param['q'], [-1])
    #     rdf = self.__radial_distortion_function(radius, q).reshape(-1, 1)
        
    #     jac = {}
    #     jac['u/w'] = self.__jac_over_w_over_u(camera, param, ud_points) # 40
    #     jac['v/w'] = self.__jac_over_w_over_u(camera, param, ud_points)
    #     jac['f(r)'] = self.__jac_over_w_over_u(camera, param, ud_points)
        
    #     pixel_points = self.__homogeneous_to_cartesian(ud_points)
        
    #     jac_x = jac['u/w'] * rdf + pixel_points[:, 0] * jac['f(r)']
    #     jac_y = jac['v/w'] * rdf + pixel_points[:, 1] * jac['f(r)']
        
    #     # https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    #     result = np.empty((jac_x.size + jac_y.size,), dtype=jac_y.dtype)
    #     result[0::2] = jac_x
    #     result[1::2] = jac_y
    #     return result

    # def jacobian(self, x : np.array, param : dict) -> np.array :
        
    #     jacobian_callbacks = {
    #         'u/w': self.__jac_over_w_over_u,
    #         'v/w': self.__jac_over_w_over_v,
    #         'f(r)': self.__jac_over_w_over_v,
    #         'f(r)/r': self.__jac_r_over_fr,
    #         'h(r)/r': self.__jac_r_over_hr,
    #         'g(r)/r': self.__jac_r_over_gr,
    #         'r': self.__jac_over_r,
    #         'u': self.__jac_over_u,
    #         'v': self.__jac_over_v,
    #         'w': self.__jac_over_w
    #     }
    #     n = param['n']
    #     q = np.reshape(param['q'], [-1])
    #     K = param['K']
    #     i = param["i"]
        
    #     # 콜백을 지역변수로 하고 메모리로 사용
    #     # 파라미터를 각 카메라 별로 분리해주는 함수가 필요
    #     # 파라미터를 
        
    #     for camera in range(n):
    #         R = param['R'][camera]
    #         t = param['t'][camera]
    #         R_matrix, _ = cv.Rodrigues(R)
    #         R_T = np.c_[R_matrix, t] # 3 x 4
            
    #         X = param["X"][camera]
    #         X_zero_one = np.c_[X, np.ones([i, 1], dtype=float)] # (40 x 4)
            
    #         camera_points = np.matmul(R_T, X_zero_one.transpose())
    #         undistorted_points = self.__homogeneous_to_cartesian(camera_points.transpose())

    #         r = self.__radius(undistorted_points)
            
    #         distorted_points = undistorted_points * np.reshape(self.__radial_distortion_function(r, q), [i, 1])
            
    #         homo_distorted_points = self.__cartesian_to_homogeneous(distorted_points)
    #         pixel_distorted_points = np.matmul(K, homo_distorted_points.transpose())
    #         distorted_points = self.__homogeneous_to_cartesian(pixel_distorted_points.transpose())
            
    #         jacobian = 2 * (distorted_points - x[camera]) * self.__jac_over_residual(camera, param, camera_points, r) # 40 x 2 x num(p)
    #         jacobian_per_image.append(jacobian)

    #     jac = np.concatenate(jacobian_per_image, axis=0)
    #     return jac
    
    # jacobian을 한번에 구할 수 있는 방법이 있다. 그러면 point 계산도 한번에 해야한다. 클래스를 다시 작성