import cv2
import click
import numpy as np
import tensorflow as tf


WINDOW_NAME = "Gesture recognition"


def calculate_temperature(raw_value: np.uint16) -> np.uint8:
    """Temperature calculation from camera raw output values.

    Parameters
    ----------
    raw_value : np.uint16
        Camera raw output value.

    Returns
    -------
    float
        Output temperature in Celsius degrees.
    """
    temperature = raw_value / 100 - 273.15

    return temperature.astype(np.uint8)


@click.command()
@click.option('-m', '--model_path', help='Path to TensorFlow SavedModel', default='./content/squeezenet_gesture_recognition_model/')
def gesture_recognition(model_path:str):
    """Gesture recognition.
    
    Notes
    -----
    SqueezeNet : https://arxiv.org/abs/1602.07360
    """

    gesture_classes = ['paper', 'rock', 'scissors', '']

    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot connect to the camera.")
            break

        frame = np.apply_along_axis(calculate_temperature, 0, frame[:120])
        mask = np.where(frame < 30, 0, 1)

        bgr_frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
        frame = np.where(np.array([mask, mask, mask]).transpose((1,2,0)), bgr_frame, 0)

        gesture_label = model.predict(np.array([frame]))
        label = gesture_classes[np.argmax(gesture_label, axis=1).astype(int)[0]]

        if len(label) > 0:
            confidence = f'{int(np.max(gesture_label) * 100)}%'

            frame = cv2.putText(frame, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, confidence, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        k = cv2.waitKey(10)

        if k == ord('q'):
            break


if __name__ == '__main__':
    gesture_recognition()
