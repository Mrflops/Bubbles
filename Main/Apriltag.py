import cv2
import numpy as np

# Initialize the detector parameters using ArUco dictionary DICT_6X6_250
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Create ArUco parameters
parameters = cv2.aruco.DetectorParameters_create()

# Load your image
image_path = 'arucocalibration.jpg'
image = cv2.imread(image_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect ArUco markers
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None and len(ids) > 0:
    # Estimate pose of each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, None, None)

    # Draw axis for each marker
    for i in range(len(ids)):
        cv2.aruco.drawAxis(image, np.eye(3), None, rvecs[i], tvecs[i], 0.1)

# Display the result
cv2.imshow('AprilTag Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
