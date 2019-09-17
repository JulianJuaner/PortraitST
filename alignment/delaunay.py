import cv2
import numpy as np
import random


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def get_triangles(subdiv, size):
    triangle_list = subdiv.getTriangleList()
    real_triangle_list = []
    r = (0, 0, size[1], size[0])
    # count = 0
    for tri in triangle_list:
        point1 = (tri[0], tri[1])
        point2 = (tri[2], tri[3])
        point3 = (tri[4], tri[5])

        if rect_contains(r, point1) and rect_contains(r, point2) and rect_contains(r, point3):
            real_triangle_list.append(tri)
    return real_triangle_list

def draw_delaunay(img, triangle_list, delaunay_color=(255, 255, 255)):
    for tri in triangle_list:
        point1 = (tri[0], tri[1])
        point2 = (tri[2], tri[3])
        point3 = (tri[4], tri[5])
        # cv2.line(img, point1, point2, delaunay_color, 1, cv2.LINE_AA, 0)
        # cv2.line(img, point2, point3, delaunay_color, 1, cv2.LINE_AA, 0)
        # cv2.line(img, point3, point1, delaunay_color, 1, cv2.LINE_AA, 0)
