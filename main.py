

from robotpy_apriltag import AprilTagPoseEstimator, AprilTagDetector
from common import *
from visualization.image_drawing import draw_tags, draw_pose
from apriltags.estimation import estimate_pose, AprilTagMap

import argparse


def main() -> None:
    # Parse runtime arguments
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--FRAME-LIMIT', help="Max number of frames to process", type=int, default=0)
    parser.add_argument('--FPS', help='Target FPS to run visualization', type=int, default=30)
    args = parser.parse_args()


    # Initiate apriltag functions and classes
    detector = AprilTagDetector()
    detector.addFamily(parameters['apriltags']['family'])
    tag_size = parameters['apriltags']['tag_size']
    apriltag_map = AprilTagMap()
    
    K = Camera.from_file('./K.txt')
    pose_estimator_config = AprilTagPoseEstimator.Config(tag_size, K.fx, K.fy, K.cx, K.cy)
    pose_estimator = AprilTagPoseEstimator(pose_estimator_config)

    video_path = glob.glob(VIDEO_FOLDER + '*.mp4')[-1]
    images = load_video(video_path, frame_limit=args.FRAME_LIMIT)

    l = len(images)
    processed_sequence = []
    print('Processing image sequence...')
    for i, image in enumerate(images):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections = detector.detect(gray_image)
        if len(detections) > 0:
            ownship_estimate, tag_poses = estimate_pose(detections, pose_estimator, apriltag_map)
            image = draw_tags(image, K, detections, tag_poses)
            image = draw_pose(image, ownship_estimate)
        processed_sequence.append(image)

        print(f'Processing: {i+1} / {l}', end='\r')
    print()
    print('Image sequence processed')


    play_frame_sequence(processed_sequence, args.FPS)



if __name__ == '__main__':
    main()