#!/usr/bin/env python3.6
# -*-coding=utf-8-*-

import mindspore_hub as mshub

from mindspore import context

from src.args import args

context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="GPU", device_id=0)

model = "noah-cvlab/gpu/1.1/ipt_v1.0_Set14_SR_x4"
network = mshub.load(model, args)

network.set_train(False)

for i in range(10):
    print(i)
