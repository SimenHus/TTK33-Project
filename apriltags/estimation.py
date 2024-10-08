
from common import Pose
from robotpy_apriltag import AprilTagDetection, AprilTagPoseEstimator


class AprilTagMap:

    def __init__(self) -> None:
        self.landmarks = {
            0: Pose()
        }


    def add_landmarks(self, new_landmarks: dict[int: Pose], detected_landmarks: dict[int: Pose]) -> None:
        for ID, T_c_new in new_landmarks.items():
            T_c_l: Pose = list(detected_landmarks.values())[0]
            T_l_new = T_c_l.inv@T_c_new
            self.landmarks[ID] = T_l_new


def estimate_pose(detections: list[AprilTagDetection], pose_estimator: AprilTagPoseEstimator, map: AprilTagMap) -> list[Pose, list[Pose]]:

    # TODO: Improve ownship calculation based on several tags. May require transformation matrices between the different tags
    tag_poses = []
    new_landmarks = {}
    detected_landmarks = {}

    ownship = None
    for detection in detections:
        pose = Pose.from_WPILib_pose(pose_estimator.estimate(detection))
        tag_poses.append(pose)


        ID = detection.getId()

        if ID == 0: ownship = pose.inv # Get ownship as the inverse of tag 0 transformation

        # if ID in map.landmarks:
        #     ownship = map.landmarks[ID]@pose.inv
        #     detected_landmarks[ID] = pose
        # if ID not in map.landmarks: new_landmarks[ID] = pose

    # Detected new landmarks and able to relate to known ones
    if len(detected_landmarks) > 0 and len(new_landmarks) > 0: map.add_landmarks(new_landmarks, detected_landmarks)

    if ownship is None: ownship = Pose()
    return ownship, tag_poses
