

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params
import copy
import torch


class Server(object):
    def __init__(self, a):
        self.A = a
        return

    def hh(self):
        return

class Client(object):
    def __init__(self, ser):
        self.serv = ser

    def cha(self):
        self.serv.A += 100
        return

ser = Server(12)
print(ser.A)

cli = Client(ser)
cli.cha()
print(ser.A)












































