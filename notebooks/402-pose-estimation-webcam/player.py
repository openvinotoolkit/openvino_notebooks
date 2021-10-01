import threading
import time

import cv2
import numpy as np


class VideoPlayer:

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            print(f"Cannot open {'camera' if isinstance(source, int) else ''} {source}")
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = \
                cv2.INTER_AREA if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH) \
                else cv2.INTER_LINEAR
        # first black frame
        self.__frame = np.zeros(
            (int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
            dtype=np.uint8
        )
        self.__lock = threading.Lock()
        self.__stop = False

    def start(self):
        self.__stop = False
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.__stop = True
        self.__cap.release()

    def run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                self.__frame = None
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

    def next(self):
        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = cv2.flip(frame, 1)
        return frame
