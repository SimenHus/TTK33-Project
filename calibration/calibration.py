import numpy as np
from os import path
import cv2
import glob
import imutils


# Define paths
BASE_FOLDER = path.join(path.curdir, 'calibration')
IMAGE_FOLDER = path.join(BASE_FOLDER, 'imgs')
OUTPUT_FOLDER = path.join(BASE_FOLDER, 'output')
IMGS = glob.glob(path.join(IMAGE_FOLDER, '*.jpg'))

visualize = False


# Define calibration parameters
boardSize = (10, 7)
squareSize = 25e-3

 
# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 


# Prepare object points
objp = np.zeros((boardSize[0]*boardSize[1], 3), np.float32)
objp[:,:2] = squareSize*np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
for fname in IMGS:
    print(path.basename(fname))
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (boardSize[0], boardSize[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == False: continue

    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    objpoints.append(objp)

    # Draw and display the corners
    cv2.drawChessboardCorners(img, (boardSize[0], boardSize[1]), corners2, ret)
    cv2.imwrite(path.join(OUTPUT_FOLDER, path.basename(fname)), img)
    if not visualize: continue
    img = imutils.resize(img, width=500) # Resize image
    cv2.imshow('img', img)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()


print('Calibrating. This may take a minute or two...', end='')
results = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)
print('Done!')

ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

np.savetxt(path.join(IMAGE_FOLDER, 'K.txt'), K)