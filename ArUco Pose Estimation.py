import cv2 as cv
import numpy as np

camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

marker_size = 0.1

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],[marker_size / 2, marker_size / 2, 0],[marker_size / 2, -marker_size / 2, 0],[-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    rvecs = []
    tvecs = []

    for c in corners:
        _, R, t = cv.solvePnP(marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)

    return rvecs, tvecs

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    if markerIds is not None:
        rvecs, tvecs = my_estimatePoseSingleMarkers(markerCorners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(markerIds)):
            result_img = cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
            print(f"Marker ID: {markerIds[i]} Position: {tvecs[i].flatten()}")

    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
