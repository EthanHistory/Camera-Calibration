import numpy as np

def initialize_camera_matrix():
    pass

def DLT(A):
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
    
    U, S, V = np.linalg.svd(A, full_matrices=True)
    print(f'<S>\n{S}')
    
    
def normalize(points):
    '''
    Normalize coordinates of points (centroid to the origin and mean distance of sqrt(2 or 3))
    
    Input
    -----
    
    Output
    -----
    
    '''
    
    points = np.asfarray(points, dtype=np.float64)
    print(f'<points>\n{points}')
    mean_, std_ = np.mean(points, axis=0), np.std(points, axis=0)
    print(f'<mean>\n{mean_}\n<std>\n{std_}')
    T = [[std_[0], 0, mean_[0]], [0, std_[1], mean_[1]], [0, 0, 1]]
    print(f'<transform>\n{T}')
    T = np.linalg.inv(T)
    homogeneous_points = np.concatenate((points.T, np.ones([1, points.shape[0]])))
    print(f'<homogeneous points>\n{homogeneous_points}')
    normalized_points = np.dot(T, homogeneous_points)
    print(normalized_points)
    return
    
if __name__ == '__main__':
    points = [[1, 2], [3, 4], [5, 6]]
    normalize(points)