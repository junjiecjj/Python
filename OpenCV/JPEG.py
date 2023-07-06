import os
import math
import imageio
import numpy as np


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    # print(f"3 (im-jm).max() = {(im-jm).max()}, (im-jm).min() = {(im-jm).min()}")
    # mse = np.mean((im * 1.0 - jm * 1.0)**2)
    mse = np.mean((im  - jm )**2)

    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr



figdir = '/home/jack/公共的/Python/OpenCV/Figures/0_3.bmp'   # bmp  jpg  jpg
savedir = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jpg'
imgio = imageio.v2.imread(figdir )

imageio.v2.imwrite(savedir, imgio, quality = 100) #  quality = 0- 100
imgio1 = imageio.v2.imread(savedir, )

size1 = os.path.getsize(figdir)
size2 = os.path.getsize(savedir)

print(f"size1 = {size1}, size2 = {size2}")


psnr = PSNR_np_simple(imgio, imgio1)
print(f"psnr = {psnr}")
# quality = 100, psnr = 41.63749305900722
# quality = 100, psnr = 16.264393311540935

