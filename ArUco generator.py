import cv2 
import numpy as np
import matplotlib.pyplot as plt

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

marker_id = 1
marker_size = 100
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

cv2.imwrite('marker_42.png', marker_image)
