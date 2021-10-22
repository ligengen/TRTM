import numpy as np
import os, sys
import pdb

def read_exr(filename):
    import OpenEXR as exr
    import Imath
    if filename[-4:] == '.pkl':
        return read_exr_from_pkl(filename)
    exrfile = exr.InputFile(filename)
    keys = exrfile.header()['channels'].keys()
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channels = []
    for c in list(keys):
        info = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        info = np.fromstring(info, dtype=np.float32)
        info = np.reshape(info, isize)
        channels.append(info)
    return channels, list(keys)


if __name__ == '__main__':
    channels, keys = read_exr('/Users/ligen/Desktop/cloth_recon/Depth_map/1/1 00.31.51/000000081.exr')
    pdb.set_trace()
