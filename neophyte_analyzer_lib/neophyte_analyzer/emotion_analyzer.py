from .face_detector import FaceDetector


class EmotionAnalyzer:

    def __init__(self):
        self.face_detector = FaceDetector()