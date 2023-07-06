#!/usr/bin/env python3.6
# -*-coding=utf-8-*-
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator


from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter


class MercatorLatitudeScale(mscale.ScaleBase):
    """
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a Mercator__ projection.

    The scale function:
      ln(tan(y) + sec(y))

    The inverse scale function:
      atan(sinh(y))

    Since the Mercator scale tends to infinity at +/- 90 degrees,
    there is user-defined threshold, above and below which nothing
    will be plotted.  This defaults to +/- 85 degrees.

    __ https://en.wikipedia.org/wiki/Mercator_projection
    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``ax.set_yscale("mercator")`` would be
    # used to select this scale.
    name = 'mercator'

    def __init__(self, axis, *, thresh=np.deg2rad(85), **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)
        if thresh >= np.pi / 2:
            raise ValueError("thresh must be less than pi/2")
        self.thresh = thresh

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.MercatorLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(
            lambda x, pos=None: f"{np.degrees(x):.0f}\N{DEGREE SIGN}")
        axis.set(major_locator=FixedLocator(np.radians(range(-90, 90, 10))),
                 major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, -self.thresh), min(vmax, self.thresh)

    class MercatorLatitudeTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the Mercator scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1 / ma.cos(masked)))
            else:
                return np.log(np.abs(np.tan(a) + 1 / np.cos(a)))

        def inverted(self):
            """
            Override this method so Matplotlib knows how to get the
            inverse transform for this transform.
            """
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(
                self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return np.arctan(np.sinh(a))

        def inverted(self):
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)


# Now that the Scale class has been defined, it must be registered so
# that Matplotlib can find it.


# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
savepath2 = '/home/jack/snap/'

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)

mscale.register_scale(MercatorLatitudeScale)
 
#===============================================================================

SNRmin = 0.0
SNRmax = 3.0
SNRinc = 0.5

SNR = np.arange(SNRmin, SNRmax+SNRinc, SNRinc)

ExtraBer = [0.1144144144,0.0317577548,0.0042500000,0.0001500000,0.0000000000,0.0000000000,0.0000000000]
ExtraFer = [0.1801801802,0.0492368291,0.0061000000,0.0002000000,0.0000000000,0.0000000000,0.0000000000]

PloadBer = [0.2183277027,0.1048144387,0.0137405208,0.0002523958,0.0000000000,0.0000000000,0.0000000000]
PloadFer = [1.0000000000,0.7355982275,0.1160000000,0.0017000000,0.0000000000,0.0000000000,0.0000000000]


fig, axs = plt.subplots(1, 1, figsize=(8, 6))
axs.grid(True)
#axs.set_xscale("log")
axs.set_yscale("log")

lb = r"$\mathrm{BER}_\mathrm{f},\ell=2$"
axs.semilogy(SNR, ExtraBer, 'r', label=lb,marker='v')

lb = r"$\mathrm{BER}_\mathrm{pf},\ell=2$"
axs.semilogy(SNR, PloadBer, 'b', label=lb,marker='s')

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
plt.xlabel(r'$\mathrm{SNR/dB}$',fontproperties=font)
plt.ylabel(r'$\mathrm{BER}$',fontproperties=font)

font1 = {'family':'Times New Roman','style':'normal','size':16}
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

 

# plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
plt.tight_layout(pad=0.5, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
#plt.subplots_adjust(top=0.89,bottom=0.01, left=0.01, right=0.99, wspace=0.4, hspace=0.2)

out_fig = plt.gcf()
out_fig.savefig(savepath2+"BER.pdf", bbox_inches = 'tight',pad_inches = 0.2)
 
plt.show()
plt.close(fig)

############################################################################################
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.grid(True)

#axs.set_xscale("log")
#axs.set_yscale("log")

lb = r"$\mathrm{WER}_\mathrm{f},\ell=2$"
axs.semilogy(SNR, ExtraFer, 'r', label=lb,marker='v')

lb = r"$\mathrm{WER}_\mathrm{pf},\ell=2$"
axs.semilogy(SNR, PloadFer, 'b', label=lb,marker='s')
#plt.yscale('mercator')


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
plt.xlabel(r'$\mathrm{SNR/dB}$',fontproperties=font)
plt.ylabel(r'$\mathrm{WER}$',fontproperties=font)

font1 = {'family':'Times New Roman','style':'normal','size':16}
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

 

# plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
plt.tight_layout(pad=0.5, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
#plt.subplots_adjust(top=0.89,bottom=0.01, left=0.01, right=0.99, wspace=0.4, hspace=0.2)

out_fig = plt.gcf()
out_fig.savefig(savepath2+"FER.pdf", bbox_inches = 'tight',pad_inches = 0.2)
 
plt.show()
plt.close(fig)










