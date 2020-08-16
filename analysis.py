#!/usr/bin/env python3

import numpy as np


def get_speeds():
    with open("data/train.txt", 'r') as f:
        d = f.read().split('\n')
        return np.array(list(map(float, d)))

s = np.loadtxt("speeds.csv")
s2 = get_speeds()[:-1]

d = s - s2

diff = np.array(list(map(abs, d)))

print(np.mean(diff))
print(np.std(diff))
