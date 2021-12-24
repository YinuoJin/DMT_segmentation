#!/usr/bin/env python

import os
import numpy as np
import argparse

from argparse import RawTextHelpFormatter


def read_skip(fid, dtype, count, skip, fsize):
    data = np.zeros(count, dtype=dtype)
    for i, c in enumerate(range(count)):
        if fid.tell() + np.dtype(dtype).itemsize >= fsize:
            break
        block= np.fromfile(fid, dtype, count=1)[0]
        data[c] = block
        fid.seek(fid.tell() + skip)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dipha edges from binary to .txt',
            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', dest='in_fname', type=str, required=True, action='store',
            help='Input dipha edges binary file')
    parser.add_argument('-o', dest='out_fname', type=str, required=True, action='store',
            help='Output dipha edges .txt file')
    
    args = parser.parse_args()
    in_fname = os.path.join(os.getcwd(), args.in_fname)
    out_fname = os.path.join(os.getcwd(), args.out_fname)
    out_path = out_fname.rpartition('/')[0]

    assert os.path.exists(in_fname), "Dipha edge binary file doesn't exist!"

    fid = open(in_fname, 'rb')
    di1, di2, n_pairs = np.fromfile(fid, np.int64, 3)
    fsize = os.path.getsize(in_fname)
    
    bverts = read_skip(fid, np.int64, n_pairs, 16, fsize)
    fid.seek(32)
    everts = read_skip(fid, np.float64, n_pairs, 16, fsize)
    fid.seek(40)
    pers = read_skip(fid, np.float64, n_pairs, 16, fsize)
    fid.close()

    vert = np.array([bverts, everts, pers]).transpose().astype(np.int64)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_fname, 'w') as ofile:
        np.savetxt(ofile, vert, fmt='%s', delimiter=' ')

