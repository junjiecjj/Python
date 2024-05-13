import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def main():
    f0_List = np.arange(100, 1001, 50)
    #print(f0)
    fs_List = np.arange(1000, 99, -50)
    #print(fs)
    t = np.arange(0, 0.0101, 0.00001)
    #print(t)
    while 1:
        a = input("键入 1 观察固定频率不同采样率时还原曲线\n\
键入 2 观察固定采样率不同频率时还原曲线\n键入 quit 退出\n")
        if a == "1":
            Change_fs(f0_List[2], fs_List, t)
        elif a == "2":
            Change_f0(f0_List, fs_List[0], t)
        elif a == "quit":
            break
        else:
            print("无法识别输入QAQ")

def Change_f0(f0_List, fs, t):
    Ts = 1 / fs
    Points_x = np.arange(-1, 1.001, Ts)#抽样点横坐标x
    plt.ion()#开启交互模式
    plt.figure(figsize=(14, 6))
    for f0 in f0_List:
        #设置画布外观
        plt.cla()
        plt.grid(True)
        plt.xlim(0, 0.01)
        plt.ylim(-1, 1)
        plt.ylabel("Amplitude")
        plt.title("fo = %ffs"%(f0/fs))
        ##设置横坐标间距
        x_major_locator=MultipleLocator(0.001)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        #设置图像线条
        F = np.cos(2*np.pi*f0*t)#原函数
        Points_y = np.cos(2*np.pi*f0*Points_x)#抽样点纵坐标y
        fa = []
        for tt in t:
            fa.append(np.dot(Points_y, np.sinc(fs*(tt-Points_x))))#抽样后还原函数
        #画图
        plt.plot(t, F, "r", label="Orignal Wave")
        plt.plot(Points_x, Points_y, "*g", label="Sampling Points")
        plt.plot(t, fa, "-.b", label="Restoration Wave")
        ##设置图例
        plt.legend(["Orignal Wave", "Sampling Points", "Restoration Wave"],\
            mode="expand", bbox_to_anchor=(0., 1.07, 1., .07), ncol=3, fancybox=True)
        plt.pause(0.4)
    plt.ioff()
    plt.show()

def Change_fs(f0, fs_List, t):
    F = np.cos(2*np.pi*f0*t)#原函数
    plt.ion()#开启交互模式
    plt.figure(figsize=(14, 6))
    for fs in fs_List:
        #设置画布外观
        plt.cla()
        plt.grid(True)
        plt.xlim(0, 0.01)
        plt.ylim(-1, 1)
        plt.ylabel("Amplitude")
        plt.title("fo = %ffs"%(f0/fs))
        ##设置横坐标间距
        x_major_locator=MultipleLocator(0.001)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        #设置图像线条
        Ts = 1 / fs
        Points_x = np.arange(-1, 1.001, Ts)
        Points_y = np.cos(2*np.pi*f0*Points_x)#抽样点
        fa = []
        for tt in t:
            fa.append(np.dot(Points_y, np.sinc(fs*(tt-Points_x))))#抽样后还原函数
        #画图
        plt.plot(t, F, "r", label="Orignal Wave")
        plt.plot(Points_x, Points_y, "*g", label="Sampling Points")
        plt.plot(t, fa, "-.b", label="Restoration Wave")
        ##设置图例
        plt.legend(["Orignal Wave", "Sampling Points", "Restoration Wave"], mode="expand", bbox_to_anchor=(0., 1.07, 1., .07), ncol=3, fancybox=True)
        plt.pause(0.4)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

