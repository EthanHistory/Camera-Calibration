import random
import glob
import cv2 as cv
assert cv.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0 but version is ' + str(cv.__version__[0])
import numpy as np
import math
import matplotlib.pyplot as plt
import radial_distortion as rd

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 50

class calibration:
    def __init__(self, pattern_size):
        self._pattern_size = pattern_size
        self._criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._objp = np.zeros((pattern_size[0] * pattern_size[1] ,3), np.float32)
        self._objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        self._image_size = ()
        self._images_path = []
        self._used_images = []

    def _random_color__(self):
        rgbl=[255,0,0]
        random.shuffle(rgbl)
        return tuple(rgbl)

    def load_images(self, image_path : str):
        self._images_path = glob.glob(image_path)

    def extract_corners(self, show=True):
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        self._used_images = []

        for fname in self._images_path:
            img = cv.imread(fname)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self._image_size = gray.shape[::-1]
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, self._pattern_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self._used_images.append(img.copy())
                objpoints.append(self._objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self._criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, self._pattern_size, corners2, ret)
                
                if show:
                    # load image using cv2....and do processing.
                    plt.title(fname)
                    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    # as opencv loads in BGR format by default, we want to show it in RGB.
                    plt.show()
        return objpoints, imgpoints
        
    def calibrate(self, objpoints : list, imgpoints : list, show=True):
        flags = cv.CALIB_RATIONAL_MODEL
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, self._image_size, None, None, flags=flags)

        for i, img in enumerate(self._used_images):
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

            # project point
            object_points = np.array([(0,0,0), (3,0,0), (0,3,0), (0,0,3)], dtype=np.float)
            image_points, jac = cv.projectPoints(object_points, rvecs[i], tvecs[i], newcameramtx, np.zeros([1, 14]))
            image_points = np.int32(image_points).reshape(-1,2)

            for img_point in image_points[1:]:
                dst = cv.line(dst, tuple(image_points[0]), tuple(img_point), self._random_color__(), 2)
            dst = cv.circle(dst, tuple(image_points[0]), 3, (0, 255, 0), -1)
            
            if show:
                plt.title(self._images_path[i])
                plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
                # as opencv loads in BGR format by default, we want to show it in RGB.
                plt.show()
        
        return ret, mtx, dist, rvecs, tvecs
    
    
class my_calibration(calibration):
    def __init__(self, pattern_size):
        super(my_calibration, self).__init__(pattern_size)
    
    def homography_test(self, objpoints, imgpoints, show=True) :
        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)
        for i in range(len(self._used_images)):
            objpoints_without_z = objpoints[i][:, :2]
            imgpoints_squeezed = imgpoints[i].squeeze()
            ret, _ = cv.findHomography(objpoints_without_z, imgpoints_squeezed)
            H = self.__get_homography__(objpoints_without_z, imgpoints_squeezed)
            print("H : ", H)
            print("ret: ", ret)

            # copy one of images
            test_image = self._used_images[i].copy()


            for point in objpoints_without_z:
                sample_point = np.append(point, 1)

                pixel_homogeneous_point_ground_truth = np.dot(ret, sample_point)
                pixel_homogeneous_point_DLT = np.dot(H, sample_point)

                pixel_point_ground_truth = [int(pixel_homogeneous_point_ground_truth[0] / pixel_homogeneous_point_ground_truth[2]),
                int(pixel_homogeneous_point_ground_truth[1] / pixel_homogeneous_point_ground_truth[2])]
                pixel_point_DLT = [int(pixel_homogeneous_point_DLT[0] / pixel_homogeneous_point_DLT[2]), int(pixel_homogeneous_point_DLT[1] / pixel_homogeneous_point_DLT[2])]
                # print("pixel point: ", pixel_point)
                test_image = cv.circle(test_image, tuple(pixel_point_ground_truth) , 5, (255, 0 , 0), -1)
                test_image = cv.circle(test_image, tuple(pixel_point_DLT) , 5, (0, 0, 255), -1)

            if show:
                # show the image
                plt.title("reprojection of tile")
                plt.imshow(cv.cvtColor(test_image, cv.COLOR_BGR2RGB))
                plt.show()

    def __get_homography__(self,objpoints, imgpoints):
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
        
        normalized_objpoints, objT = self.__normalize__(objpoints)
        normalized_imgpoints, imgT = self.__normalize__(imgpoints)
        
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
    
    def __normalize__(self, points):
        '''
        Normalize coordinates of points (centroid to the origin and mean distance of sqrt(2 or 3))
        
        Input
        -----
        
        Output
        -----
        
        '''
        
        points = np.asfarray(points, dtype=np.float64)
        mean_, std_ = np.mean(points, axis=0), np.std(points, axis=0)
        T = [[std_[0], 0, mean_[0]], [0, std_[1], mean_[1]], [0, 0, 1]]
        T = np.linalg.inv(T)
        homogeneous_points = np.concatenate((points.T, np.ones([1, points.shape[0]])))
        normalized_points = np.dot(T, homogeneous_points).T[:, :2]
        return normalized_points, T

    def initial_intrinsic_extrinsic_test(self, objpoints, imgpoints) :
        list_of_H = []

        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)

        for i in range(len(self._used_images)):
            objpoints_without_z = objpoints[i][:, :2]
            imgpoints_squeezed = imgpoints[i].squeeze()
            H = self.__get_homography__(objpoints_without_z, imgpoints_squeezed)
            list_of_H.append(H)

        B = self.__DLT_for_B__(list_of_H)
        print(f'B = {B}\n')
        K = self.__extract_intrinsic__(B).reshape([3,3])
        print(f'\nK = {K}')

        rvecs, tvecs = self.__extract_extrinsic__(K, list_of_H)

        return K, rvecs, tvecs

    def __DLT_for_B__(self, list_of_H:list) -> np.ndarray:
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

        for H in list_of_H:
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

    def __extract_intrinsic__(self, B : np.array) -> np.array :
        # B = [B11, B12, B22, B13, B23, B33]^T
        assert(B.shape == (6,))

        # result of Cholesky decomposition
        v0 = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1] * B[1])
        lambda_ = B[5] - (B[3] * B[3] + v0 * (B[1] * B[3] - B[0] * B[4])) / B[0]
        alpha_ = math.sqrt(lambda_ / B[0])
        beta_ = math.sqrt(lambda_ * B[0] / (B[0] * B[2] - B[1] * B[1]))
        gamma_ = -B[1] * alpha_ * alpha_ * beta_ / lambda_
        u0 = gamma_ * v0 / beta_ - B[3] * alpha_ * alpha_ / lambda_

        print(f'v0 = {v0}')
        print(f'lambda_ = {lambda_}')
        print(f'alpha_ = {alpha_}')
        print(f'beta_ = {beta_}')
        print(f'gamma_ = {gamma_}')
        K = np.array([alpha_, gamma_, u0, 0, beta_, v0, 0, 0, 1])


        return K

    # def __K_dict__(self, K : np.array) :
    #     K_ = K.reshape(9)
    #     return dict(fx=K_[0], skew=K_[1], px=K_[2], fy=K_[4], py=K_[5])
    
    def __extract_extrinsic__(self, K : np.array, Hs : list) -> np.array :

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

    def radial_distortion_test(self, objpoints, imgpoints, show=True):
        ret, mtx, dist, rvecs, tvecs = self.calibrate(objpoints, imgpoints, False)
        mtx_np = np.array(mtx)
        dist_np = np.array(dist)
        rvecs_np = np.array(rvecs)
        tvecs_np = np.array(tvecs)
        objpoints_np = np.array(objpoints)

        p = np.array([[2.31556927e+01, -2.56866259e+01, -5.11172497e+00,  2.36050153e+01, -1.70086701e+01, -1.64143823e+01]])
        param = rd.params_dict(13, 40, mtx_np, p, rvecs_np, tvecs_np, objpoints_np)
        
        # rdm = rd.radial_distortion()
        rdm = rd.radial_distortion_test()
        
        distorted_points = rdm.radial_distortion_model(param)

        for i, img in enumerate(self._used_images):
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = img.copy()

            for points in distorted_points[i]:
                dst = cv.circle(dst, (int(points[0]), int(points[1])), 3, (0, 255, 0), -1)
            
            if show:
                plt.title(f'radial distortion test {i}')
                plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
                # as opencv loads in BGR format by default, we want to show it in RGB.
                plt.show()