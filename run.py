#!/usr/bin/env python3

from model import Model
from img_proc import ImageProc
from image_util import ImageUtil


# load pre-processed data

iutil = ImageUtil()

#frames, speeds = iutil.gen_data("data/train.mp4", "data/train.txt")
frames, speeds = iutil.load_data()

# train model

model = Model()

model.train(frames, speeds, 50, 32, 0.1)

model.test(frames, speeds)

