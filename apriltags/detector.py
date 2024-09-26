import cv2
import numpy as np
from dt_apriltags import Detector


class DetectorClass(Detector): 

    def __init__(self, K=np.eye(3), tagSize=1, **kwargs):
        super().__init__(**kwargs)


        self.cameraCalibrationParams = (K[0,0], K[1,1],
                                        K[0,2], K[1,2])
        self.tagSize = tagSize
        self.K = K
        

    def detect(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert image to grayscale
        tags = super().detect(grayImg, estimate_tag_pose=True,
                              camera_params=self.cameraCalibrationParams, tag_size=self.tagSize)
        
        return tags
    

class DetectorClass()