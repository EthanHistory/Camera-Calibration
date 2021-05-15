import sys
import numpy as np
import calibration as ca
import radial_distortion as rd

print(sys.executable)

tests = ['calibration', 'homography', 'init_cam_param', 'my_calibration']

if __name__ == '__main__':
    pattern_size = (8, 5)
    
    for test in tests:
        if test == 'calibration':
            calibration_object = ca.calibration(pattern_size)
            calibration_object.load_images('test_images/*.jpg')
            objpoints, imgpoints = calibration_object.extract_corners(False)
            ret, mtx, dist, rvecs, tvecs = calibration_object.calibrate(objpoints, imgpoints, False)
            print(f'reprojection error (RMS) : {ret}')
            print(f'camera matrix : {mtx}')
            print(f'distortion coeffcients : {dist}')

        if test == 'homography':
            my_calibration_object = ca.my_calibration(pattern_size)
            my_calibration_object.load_images('test_images/Ladybug2.jpg')
            objpoints, imgpoints = my_calibration_object.extract_corners(False)
            my_calibration_object.homography_test(objpoints, imgpoints, False)
            
        if test == 'init_cam_param':
            my_calibration_object.load_images('test_images/*.jpg')
            objpoints, imgpoints = my_calibration_object.extract_corners(False)
            
            K, rvecs_, tvecs_ = my_calibration_object.initial_intrinsic_extrinsic_test(objpoints, imgpoints)
            # ground_truth_K = np.array([[548.88848966, 0., 501.21329994], [0., 547.20840059, 390.6931097 ], [0., 0., 1.]])
            # print(f'\nground truth K = {ground_truth_K}')
            # print(f'\nDiff = {K - ground_truth_K}')
            ground_K = mtx
            print(f'\nground truth K = {ground_K}')
            print(f'\nDiff = {K - ground_K}')

            ground_rvecs = np.array(rvecs)
            ground_tvecs = np.array(tvecs)

            for i in range(len(rvecs_)):
                print(f'\n[image 1]---------------------------')
                print(f'\nrvec Diff = {rvecs_[i] - ground_rvecs[i]}')
                print(f'\ntvec Diff = {tvecs_[i] - ground_tvecs[i]}')
                
            
        if test == 'my_calibration':
            my_calibration_object.load_images('test_images/*.jpg')
            objpoints, imgpoints = my_calibration_object.extract_corners(False)
            my_calibration_object.radial_distortion_test(objpoints, imgpoints)