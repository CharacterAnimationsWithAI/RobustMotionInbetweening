import os
import torch
import torch.nn as nn
import numpy as np
from skeleton.animation_data import AnimationData
from skeleton import bvh


def PLU(x, alpha = 0.1, c = 1.0):
    relu = nn.ReLU()
    o1 = alpha * (x + c) - c
    o2 = alpha * (x - c) + c
    o3 = x - relu(x - o2)
    o4 = relu(o1 - o3) + o3
    return o4


def gen_ztta(timesteps=60, dim=256):
    ztta = np.zeros((timesteps, dim))
    for t in range(timesteps):
        for d in range(dim):
            if d % 2 == 0:
                ztta[t, d] = np.sin(t / (10000 ** (d // dim)))
            else:
                ztta[t, d] = np.cos(t / (10000 ** (d // dim)))
    return torch.from_numpy(ztta.astype(float))


def save_bvh_from_network_output(rotations, output_path):
    anim = AnimationData.from_network_output(rotations)
    bvh, names, ftime = anim.get_BVH()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    bvh.save(output_path, bvh, names, ftime)
