import glob
import json
import numpy as np
import cv2
import time

from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field

DATA_FOLDER = './data/'
IMAGE_FOLDER = DATA_FOLDER + 'images/'
FPS = 30

# Load parameters from yaml file
with open('./parameters.json', 'r') as f: parameters = json.load(f)

### --------------- FUNCTIONS ---------------
def load_images(max_frames=0) -> list:
    image_paths = glob.glob(IMAGE_FOLDER+'*.jpg')
    max_frames = len(image_paths) if max_frames == 0 else max_frames

    frames = []
    for i, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path) # Read image from file
        frames.append(frame)
        if i >= max_frames: break

    return frames


def timeit(func):
    def wrapper(*args, **kwargs): 
        start = time.time() 
        result = func(*args, **kwargs) 
        end = time.time() 
        print(f'Successfully executed {func.__name__} in {end-start:.2f} seconds')
        return result 
    return wrapper



def play_frame_sequence(frames: list) -> None:
    started = False
    for frame in frames:
        frame = cv2.resize(frame, (0, 0), fx = 0.3, fy = 0.3)
        cv2.imshow('Video', frame)

        if not started:
            print('Playback ready, press any key to begin...')
            cv2.waitKey(0)
            started = True

        if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


### --------------- CLASSES ---------------
@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    _K: 'np.ndarray[3, 3]' = field(default_factory=lambda: np.eye(3))

    @staticmethod
    def from_file(path: str) -> 'Camera':
        # Load camera intrinsic parameters
        params = np.loadtxt(path).reshape((3,3))

        return Camera(
            params[0, 0],
            params[1, 1],
            params[0, 2],
            params[1, 2],
            params
        )

    def __matmul__(self, other):
        return self._K@other
    
    def __repr__(self):
        return repr(self._K)


@dataclass
class Pose:
    R: 'np.array[3, 3]' = field(default_factory=lambda: np.eye(3))
    t: 'np.array[3]' = field(default_factory=lambda: np.zeros((3,)))

    def __post_init__(self):
        if self.R.shape[0] > 3:
            self.T = self.R
            self.R = self.T[:3, :3]
            self.t = self.T[:3, 3]
        else:
            self.T = np.zeros((4, 4))
            self.T[:3, :3] = self.R
            self.T[:3, 3] = self.t
            self.T[3, 3] = 1

    @property
    def inv(self):
        RT = self.R.T
        T = Pose(RT, -RT@self.t) # Faster processing than matrix inversion
        return T

    @property
    def pos(self):
        return self.t
    
    @property
    def rot(self):
        return self.R

    def __repr__(self):
        return repr(self.T)

    def __matmul__(self, other):
        return Pose(self.T@other.T)
    
    @staticmethod
    def from_WPILib_pose(pose) -> 'Pose':
        R = Rotation.from_rotvec(pose.rotation().axis()).as_matrix()
        t = np.array((pose.X(), pose.Y(), pose.Z()))

        return Pose(R, t)