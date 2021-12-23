from .face_detector import FaceDetector
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from deepface.basemodels import VGGFace

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def create_model(input_shape, num_classes):
    data_augmentation = keras.Sequential(
        [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        ]
    )
    vgg = VGGFace.baseModel()

    cnt = 0
    for lyer in vgg.layers:
        if cnt > 34:
            lyer.trainable = True
            lyer.activation = tf.keras.layers.Activation('tanh')
        else:
            lyer.trainable = False
        cnt = cnt + 1
    vgg_face = Model(inputs=vgg.layers[0].input, outputs=vgg.layers[-2].output)

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = vgg_face(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

class EmotionAnalyzer:

    def __init__(self, input_shape = (224, 224), num_classes = 7, weights_dir='ml_models'):
        self.face_detector = FaceDetector()
        self.model = create_model(input_shape = input_shape, num_classes = num_classes)
        self.model.load_weights(f'{weights_dir}/save_at_40 (1).h5')

    def predict(self, image):
        if self.face_detector.detect_face(image):
            prep_face = self.face_detector.post_process_face()
            nn_prediction = np.array(self.model.predict(prep_face)).argmax()
            return (nn_prediction, emotions[nn_prediction])
        else:
            return (-1, 'Face isn`t detected')

    def final_desision(self, ts_data):
        res = np.mean(ts_data)
        return res > 3 and res < 6