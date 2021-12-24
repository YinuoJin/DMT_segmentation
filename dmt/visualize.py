#!/usr/bin/env python

import os
import numpy as np
import argparse

from skimage import io
from argparse import RawTextHelpFormatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Discrete Morse graph into 2D / 3D mask',
                                      formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', dest='fname', type=str, required=True, action='store',
                        help='Input discrete morse graph file name')
    parser.add_argument('-m', dest='mask', type=str, required=True, action='store',
                        help='Output mask file name')
    parser.add_argument('-d', dest='dim', type=int, nargs='+', required=True, action='store',
                        help='Output mask dimension')
    args = parser.parse_args()

    fname = os.path.join(os.getcwd(), args.fname)
    mask_name = args.mask
    dim = tuple(args.dim)
    assert os.path.exists(fname), "DMT vert output file doesn't exist"

    vert_out = np.loadtxt(fname).astype(np.int64)
    mask = np.zeros(dim)

    for i in range(len(vert_out)):
        mask[vert_out[i, 0], vert_out[i, 1], vert_out[i, 2]] = 1
    io.imsave(mask_name, mask)


