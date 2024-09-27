import numpy as np
import cv2
from common import Pose, Camera
from robotpy_apriltag import AprilTagDetection

font_face = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.8
line_thickness = 3
outline_color = (0, 255, 0)
text_color = (255, 0, 0)

def draw_tags(image: 'np.ndarray', K: Camera, detections: list[AprilTagDetection], tag_poses: list[Pose]) -> 'np.ndarray':
    
    for detection, pose in zip(detections, tag_poses):
        # Draw contour of tag
        for i in range(4):
            j = (i + 1) % 4
            point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
            point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
            cv2.line(image, point1, point2, outline_color, line_thickness)

        # Add tag id next to tag
        cv2.putText(image, str(detection.getId()),
                    org=(int(detection.getCorner(0).x)+10, int(detection.getCorner(0).y)+10),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=text_color)

        
        # Draw coordinate frame of tag
        scale = 0.1
        X = pose.T @ np.array([
            [0,scale,0,0],
            [0,0,scale,0],
            [0,0,0,scale],
            [1,1,1,1]])
        u, v = project(K, X)
        u, v = u.astype(int), v.astype(int)
        cv2.line(image, [u[0], v[0]], [u[1], v[1]], color=(255, 0, 0), thickness=line_thickness)
        cv2.line(image, [u[0], v[0]], [u[2], v[2]], color=(0, 255, 0), thickness=line_thickness)
        cv2.line(image, [u[0], v[0]], [u[3], v[3]], color=(0, 0, 255), thickness=line_thickness)
        cv2.putText(image, 'X', (u[1], v[1]), font_face, font_scale, (255, 255, 255))
        cv2.putText(image, 'Y', (u[2], v[2]), font_face, font_scale, (255, 255, 255))
        cv2.putText(image, 'Z', (u[3], v[3]), font_face, font_scale, (255, 255, 255))

    return image


def draw_pose(image: 'np.ndarray', pose: Pose) -> 'np.ndarray':
    font_scale = 3
    thickness = 4
    y_start = 100
    y_spacing = 100
    cv2.putText(image, f'x: {pose.x:.2f}', (50, y_start), font_face, font_scale, (255, 0, 0), thickness)
    cv2.putText(image, f'y: {pose.y:.2f}', (50, y_start + y_spacing), font_face, font_scale, (255, 0, 0), thickness)
    cv2.putText(image, f'z: {pose.z:.2f}', (50, y_start + 2*y_spacing), font_face, font_scale, (255, 0, 0), thickness)

    return image


def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]