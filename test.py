#!/usr/bin/env python3.6
#-*-coding=utf-8-*-

# from ipyvolume import p3


# fig = p3.figure()
# p3.style.use('dark')

# s = p3.quiver(*ds_stream.data,size=6)
# p3.animate_glyphs(s,interval = 200)
# p3.show()



import ipyvolume

hzd = ipyvolume.datasets.hdz2000.fetch()

ipyvolume.volshow(hzd.data,lighting = True,width=300,height=300,level=[0.4,0.6,0.9])