import os
import cv2
import numpy as np


PAPER_KEY = 'p'
STONE_KEY = 'k'
SCISSORS_KEY = 'n'
OTHER_KEY = 'o'

WINDOW_NAME = "Gesture capture"


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


def gesture_capture():
    """Camera capture for gesture dataset."""

    for data_class in ['paper', 'stone', 'scissors', 'other']:
        if not os.path.exists(f'dataset/{data_class}'):
            os.makedirs(f'dataset/{data_class}')

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot connect to the camera.")
            break

        frame = np.apply_along_axis(calculate_temperature, 0, frame)
        mask = np.where(frame < 30, 0, 1)

        bgr_frame = cv2.cvtColor(np.uint8(frame),cv2.COLOR_GRAY2BGR)
        frame = np.where(np.array([mask, mask, mask]).transpose((1,2,0)), bgr_frame, 0)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10)

        if key == ord('q'):
            break
        elif key == ord(PAPER_KEY):
            num = len(os.listdir('dataset/paper'))
            cv2.imwrite(f'dataset/paper/{str(num).zfill(3)}.jpg', frame)
        elif key == ord(STONE_KEY):
            num = len(os.listdir('dataset/stone'))
            cv2.imwrite(f'dataset/stone/{str(num).zfill(3)}.jpg', frame)
        elif key == ord(SCISSORS_KEY):
            num = len(os.listdir('dataset/scissors'))
            cv2.imwrite(f'dataset/scissors/{str(num).zfill(3)}.jpg', frame)
        elif key == ord(OTHER_KEY):
            num = len(os.listdir('dataset/other'))
            cv2.imwrite(f'dataset/other/{str(num).zfill(3)}.jpg', frame)


if __name__ == '__main__':
    gesture_capture()
