#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from model import Model
from img_proc import ImageProc
from image_util import ImageUtil


# load pre-processed data

iutil = ImageUtil()

frames = iutil.get_frames("data/test.mp4")

# train model

model = Model()

model.load_weights("model1.h5")

pred = model.predict(frames)

pred = np.append(pred, pred[-1])

print(len(pred))

np.savetxt("pred.txt", pred, fmt='%f')

print("=" * 60)
print("MIN: {}".format(np.min(pred)))
print("MAX: {}".format(np.max(pred)))
print("MEAN: {}".format(np.mean(pred)))
print("STD: {}".format(np.std(pred)))
print("=" * 60)

#print(pred)

f = [i for i in range(len(pred))]

plt.plot(f, pred, 'gx')

plt.xlabel('Frame', fontsize=16)
plt.ylabel('Speed', fontsize=16)

plt.show()
