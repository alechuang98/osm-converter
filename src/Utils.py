import numpy as np
import math
import Constant

def outer(a, b):
    return a[0] * b[1] - a[1] * b[0]

def distance2(a, b):
    return np.sum((a - b) * (a - b))

def distance(a, b):
    return math.sqrt(distance2(a, b))

def get_line_intersection(l1, l2):
    p1 = np.array(l1[0])
    p2 = np.array(l1[1])
    q1 = np.array(l2[0])
    q2 = np.array(l2[1])
    if abs(outer(p2 - p1, q2 - q1)) < Constant.EPS and abs(outer(p2 - p1, q1 - p1)) < Constant.EPS:
        return l2[0]
    f1 = outer(p2 - p1, q1 - p1)
    f2 = outer(p2 - p1, p1 - q2)
    f = f1 + f2
    assert abs(f) > Constant.EPS
    return q1 * (f2 / f) + q2 * (f1 / f)

def get_circle_and_segment_intersection(l, center, r):
    """
    one of l[0], l[1] should be inside and outside
    """
    inside, outside = l[0], l[1]
    if distance(center, inside) > distance(center, outside):
        inside, outside = outside, inside
    for _ in range(64):
        mid = (inside + outside) / 2
        if distance(center, mid) < r:
            inside = mid
        else:
            outside = mid
    return inside

def rotate(a, theta, center):
    rotate_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    return (rotate_matrix @ (a - center)) + center

