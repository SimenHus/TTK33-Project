

from robotpy_apriltag import AprilTagPoseEstimator, AprilTagDetector
from common import *
from figure import draw_tag


def main() -> None:
    detector = AprilTagDetector()
    detector.addFamily(parameters['apriltags']['family'])
    tag_size = parameters['apriltags']['tag_size']

    K = Camera.from_file('./K.txt')
    pose_estimator_config = AprilTagPoseEstimator.Config(tag_size, K.fx, K.fy, K.cx, K.cy)
    pose_estimator = AprilTagPoseEstimator(pose_estimator_config)

    images = load_images()
    img = images[0]
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detection = detector.detect(gray_image)


    draw_tag(img, K, detection, pose_estimator)

    play_frame_sequence([img])



if __name__ == '__main__':
    main()