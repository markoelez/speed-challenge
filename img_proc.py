#!/usr/bin/env python3

import cv2
import numpy as np


class ImageProc:

    def optical_flow(self, frame1, frame2, size):
        f1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        hsv = np.zeros((size[0], size[1], 3))
        
        # increase saturation
        hsv[..., 1] = 255

        # dense
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 1, 15, 2, 5, 1.3, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 2] = (mag * 20).astype(int)

        return np.array(hsv)

    def crop_resize(self, frame, size):
        #fc = frame[23 : 375, :]
        fc = frame[200:400]

        f = cv2.resize(fc, size, interpolation=cv2.INTER_AREA)

        return f

    def process_frames(self, frame1, frame2, size):
        f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        f2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        f1 = self.crop_resize(f1, size)
        f2 = self.crop_resize(f2, size)

        flow = self.optical_flow(f1, f2, size)

        return flow
