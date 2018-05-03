import numpy as np
import pandas as pd

def rotate (origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    
    try:
        px, py = points[:,0], points[:,1]
    except TypeError:
        px, py = points[0], points[1]
        
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    
    return qx, qy

def procrustes(p, angle, ma_window=12):
    """
    Normalize a set of 2D landmarks by translating, scaling and rotating.

    Set p of n-points expected to be n-by-2. Angle should be given in radians.

    """

    # Smooth by moving average if a window is provided
    if ma_window:
        
        s = pd.rolling_mean(p, ma_window)
        s[0:12, :] = p[0:12, :]
        p = s

    # Translate shape to center at origin by subtracting the mean of each shape
    p -= np.mean(p, axis=0)

    # Scaling by root mean squared of shape
    p /= np.sum(np.power(p, 2))
    
    # Rotation by provided angle
    px, py = rotate( (0, 0), p, angle)

    return np.transpose(np.vstack((px, py)))

def mean_shape(shapes):
    """
    Calculate mean 2D shape of a set of sets of points.

    Expecting an n-by-2*k array for n points and k sets of points
    """

    x_coords = np.vstack(shapes[:, 0::2])
    y_coords = np.vstack(shapes[:, 1::2])

    return np.vstack( (np.mean(x_coords, axis=1), np.mean(y_coords, axis=1)) )

def align(p, landmark1, landmark2, left, right):
    """
    Align pedestal to landmarks given a left/right anchor
    """

    x1, y1 = landmark1[:,0], landmark1[:,1]
    x2, y2 = landmark2[:,0], landmark2[:,1]

    m1 = (y2 - y1)/(x2 - x1)
    b1 = y1 - m1*x1

    m2 = -1/m1
    b2 = p[:, 1::2] - p[:,0::2]*m2

    x_int = (b2 - b1)/(m1 - m2)
    y_int = m2*x_int + b2


