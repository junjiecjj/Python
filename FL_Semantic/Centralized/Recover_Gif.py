#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:44:20 2023

@author: jack
"""



from PIL import Image
# import ImageDraw
import imageio
import os
import glob



def gif_maker(R = 0.1, trainSNR = None, rootdir = "/home/jack/SemanticNoise_AdversarialAttack/MakeGif/R_noiseless/"):
    image_list = []
    # filename = rootdir + f"R={R:.1f}/raw_grid_images.png"
    # im = Image.open(filename)
    # print(f"{im.size}")
    # image_list.append(im)

    filelist = []
    snrlist = [-2, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 30, 35, 40]
    for snr in snrlist:
        if trainSNR == None:
            filelist.append(os.path.join(rootdir, f"R={R:.1f}", f"grid_images_R={R:.1f}_trainSnr=noiseless(dB)_testSnr={snr}(dB).png"))
        else:
            filelist.append(os.path.join(rootdir, f"R={R:.1f}_trainSNR={trainSNR}(dB)", f"grid_images_R={R:.1f}_trainSnr={trainSNR}(dB)_testSnr={snr}(dB).png"))

    for i, file in enumerate(filelist):
        # print(f"{i} = {file}\n")
        im = Image.open(file)
        # print(f"{i} {im.size}\n")
        image_list.append(im)

    if trainSNR == None:
        imageio.mimsave(os.path.join(rootdir, f"R={R:.1f}", f"R={R:.1f}_animation.gif"), image_list, 'GIF',  duration = 1000)
    else:
        imageio.mimsave(os.path.join(rootdir, f"R={R:.1f}_trainSNR={trainSNR}(dB)", f"R={R:.1f}_trainSNR={trainSNR}(dB)_animation.gif"), image_list, 'GIF',  duration = 1000)
    return


gif_maker(R = 0.7,  trainSNR = 10, rootdir = "/home/jack/SemanticNoise_AdversarialAttack/MakeGif/R_SNR/")

# gif_maker(R = 0.1, rootdir = "/home/jack/SemanticNoise_AdversarialAttack/MakeGif/R_noiseless/")

# image_list = []
# for i in range(1, 11):
#     filename = 'image_' + str(i) + '.png'
#     im = Image.open(filename)
#     image_list.append(im)


# imageio.mimsave('animation.gif', image_list)

