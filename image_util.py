#!/usr/bin/env python3

import os
import shutil
import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm
from img_proc import ImageProc


class ImageUtil:

    def __init__(self):
        self.proc = ImageProc()

        self.DSIZE = (100, 100)

        self.output_dir = "frames/"

    def load_speeds(self, fn):
        with open(fn, 'r') as f:
            d = f.read().split('\n')
            return np.array(list(map(float, d)))

    def load_vf(self, fn):
        v = cv2.VideoCapture(fn)
        v.set(cv2.CAP_PROP_FPS, 20)
        d = []
        while 1:
            ret, f = v.read()
            if not ret: break
            d.append(np.array(f))
        return np.array(d)

    def load_vf_sk(self, fn):
        return skvideo.io.vread(fn)

    def reset_dirs(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)

    def gen_data(self, vid_file, speed_file):
        speeds = self.load_speeds(speed_file)
        vid = self.load_vf_sk(vid_file)

        assert(len(vid) == len(speeds))

        self.reset_dirs()

        mean_speeds, frames = [], []

        for i in tqdm(range(len(vid) - 1)):
            idx1 = i
            idx2 = i + 1

            f1 = vid[idx1]
            f2 = vid[idx2]

            speed1 = speeds[idx1]
            speed2 = speeds[idx2]

            flow = self.proc.process_frames(f1, f2, self.DSIZE)

            path = "{}/{}.png".format(self.output_dir, i)

            cv2.imwrite(path, flow)

            mean_speeds.append(np.mean([speed1, speed2]))
            frames.append(flow)

        np.savetxt("speeds.csv", mean_speeds)

        return (np.array(frames), np.array(mean_speeds))

    def load_data(self):
        speeds = np.loadtxt("speeds.csv")
        frames = []

        frame_paths = os.listdir(self.output_dir)
        for fn in tqdm(frame_paths):
            frames.append(cv2.imread(self.output_dir + fn))

        return (np.array(frames), speeds)

    def play(self, frames):
        for f in frames:
            cv2.imshow('window', f)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    util = ImageUtil()

    frames, speeds = util.gen_data("data/train.mp4", "data/train.txt")
    #frames, speeds = util.load_data()

    #frames = util.load_vf_sk("data/train.mp4")
    #frames = [util.proc.crop_resize(f, (500, 500)) for f in frames]

    util.play(frames)

