#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from model import Model
from img_proc import ImageProc
from image_util import ImageUtil


# load pre-processed data

iutil = ImageUtil()

#frames, speeds = iutil.gen_data("data/train.mp4", "data/train.txt", use_mean=True)
frames = iutil.get_frames("data/test.mp4")
#frames, speeds = iutil.load_data()

# train model

model = Model()

#model.train(frames, speeds, 150, 32, 0.1)

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

#print(model.test(frames, speeds))

f = [i for i in range(len(pred))]

plt.plot(f, pred, 'gx')
#plt.plot(f, avg, 'r')#color='#f69f7c')

plt.xlabel('Frame', fontsize=16)
plt.ylabel('Speed', fontsize=16)

plt.show()
