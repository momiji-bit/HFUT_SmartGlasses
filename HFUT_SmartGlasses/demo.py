import time

import cv2
import multiprocessing


def my_videocapture(cap_num):
    cap = cv2.VideoCapture(cap_num)
    while True:
        ret0, frame = cap.read()
        cv2.imshow("Cam{}_Demo".format(cap_num), frame)
        cv2.waitKey(1)
        # time.sleep(0.5)

if __name__ == '__main__':
    cam_l_process = multiprocessing.Process(target=my_videocapture, args=(0,))
    cam_r_process = multiprocessing.Process(target=my_videocapture, args=(1,))

    cam_l_process.start()
    cam_r_process.start()