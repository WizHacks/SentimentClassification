import math
import scipy.spatial.distance as scipy_dist

def euclidean(xi, xj):
    '''Calculate euclidean distance between xi and xj'''
    distance = 0
    for m in range(len(xi)):
        distance += pow((xi[m] - xj[m]), 2)
    return math.sqrt(distance)

def manhattan(xi, xj):
    '''Calculate manhattan distance between xi and xj'''    
    distance = 0
    for m in range(len(xi)):
        distance += abs(xi[m] - xj[m])
    return distance

def euclidean_scipy(xi, xj, p=2):
    '''Calculate euclidean distance between xi and xj'''    
    return scipy_dist.euclidean(xi, xj)

def manhattan_scipy(xi, xj, p=1):
    '''Calculate manhattan distance between xi and xj'''    
    return scipy_dist.cityblock(xi, xj)

def minkowski_scipy(xi, xj, p):
    '''Calculate manhattan distance between xi and xj'''    
    return scipy_dist.minkowski(xi, xj, p)