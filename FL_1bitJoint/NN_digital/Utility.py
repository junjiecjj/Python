
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""


import scipy
import numpy as np
import torch
import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



# 初始化随机数种子
def set_random_seed(seed = 42,):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def set_printoption(precision = 3):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(
        precision = precision,    # 精度，保留小数点后几位，默认4
        threshold = 1000,
        edgeitems = 3,
        linewidth = 150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile = None,
        sci_mode = False  # 用科学技术法显示数据，默认True
    )


def mess_stastic(message_lst, D, args, comm_round, savedir, ):
    # allmes = np.array(message_lst).flatten()
    allmes = np.zeros((len(message_lst), D))
    for i, param_W in enumerate(message_lst):
        params_float = torch.Tensor()
        for key, val in param_W.items():
            if key != "conv2_norm.running_mean" and key != "conv2_norm.running_var" and key != "conv2_norm.num_batches_tracked":
                params_float = torch.cat((params_float, val.detach().cpu().flatten()))
        allmes[i,:] = np.array(params_float)
        # break
    # print(f"Done!!!!!!!!")
    fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
    print(f"allmes = {allmes.shape}")
    lb = f"Round={comm_round}"
    count, bins, ignored = axs.hist(allmes.flatten(), density=False, bins='auto', histtype='stepfilled', alpha=0.5, facecolor = "#0099FF", label= lb, zorder = 4)

    # mu  = allmes.mean()
    # std = allmes.std()
    # X = np.linspace(allmes.min(), allmes.max(), 100)
    # N_pdf = scipy.stats.norm.pdf(X, loc = mu, scale = std)
    # axs.plot(X, N_pdf, c = 'b', lw = 2, label = f"N({mu}, {std**2})")

    ## 直方图和核密度估计图
    # sns.histplot(allmes, kde = True,  ax=axs, stat = "density", bins = 50, color='skyblue')
    # axs.axvline(x=mu, color = 'r', linestyle = '--')
    # axs.axvline(x=mu + std, color = 'r', linestyle = '--')
    # axs.axvline(x=mu - std, color = 'r', linestyle = '--')

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
    # legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    # frame1 = legend1.get_frame()
    # frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
    axs.set_xlabel('Value', fontdict = font, )
    axs.set_ylabel('Density', fontdict = font, )
    axs.set_title(f"{lb}", fontdict = font,  )

    axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

    # 显示图形
    out_fig = plt.gcf()
    out_fig.savefig(savedir + f'/round_{comm_round}.eps', pad_inches = 0,)
    plt.show()
    return






































































































































































