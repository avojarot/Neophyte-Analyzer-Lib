import numpy as np
import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, img_size=224):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                                  min_detection_confidence=0.5)
        self.image = []
        self.img_size = img_size

    def detect_face(self, image):
        self.image = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.detector.process(image)
        if results.detections:
            self.image = results.detections[0]
        return bool(results.detections)

    def post_process_face(self):
        if not self.image:
            raise ReferenceError('Face isn`t detected')

        im = cv2.resize(self.image, (self.img_size, self.img_size))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(im)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        img2 = img2.reshape(-1, self.img_size, self.img_size, 3)
        self.image = img2



