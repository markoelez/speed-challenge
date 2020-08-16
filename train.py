#!/usr/bin/env python3

import numpy as np
from model import Model
from img_proc import ImageProc
from image_util import ImageUtil


# load pre-processed data

iutil = ImageUtil()

frames, speeds = iutil.gen_data("data/train.mp4", "data/train.txt", use_mean=True)

model = Model()

model.train(frames, speeds, 150, 32, 0.1)

