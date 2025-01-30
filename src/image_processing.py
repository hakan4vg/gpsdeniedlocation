import cv2
import numpy as np
import requests
from typing import Tuple

class ImageProcessor:
    def __init__(self, config):
        """
        Initializes the ImageProcessor with the given configuration.
        """
        self.config = config
        algorithm = config['feature_extraction']['algorithm']
        params = config['feature_extraction']['params']

        if algorithm == 'ORB':
            self.feature_extractor = cv2.ORB_create(
                        nfeatures=params['nfeatures'],
                        scaleFactor=params['scaleFactor'],
                        nlevels=params['nlevels'])
        else:
            raise NotImplementedError(f"Feature extraction algorithm '{algorithm}' not implemented")


    def extract_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Extracts relevant features from the input image for localization.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        return keypoints, descriptors
        

    def get_image(self):
        """
        Captures an image from the drone camera OR synthetic.
        """
        # Placeholder image (e.g., load from file for testing)
        test_image = cv2.imread("synthetic_test_image.png")
        return test_image
