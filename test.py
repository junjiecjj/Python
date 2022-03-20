#!/usr/bin/env python3.6
#-*-coding=utf-8-*-

# from ipyvolume import p3


# fig = p3.figure()
# p3.style.use('dark')

# s = p3.quiver(*ds_stream.data,size=6)
# p3.animate_glyphs(s,interval = 200)
# p3.show()



import torch
import torch.utils.data as Data
 
BATCH_SIZE = 5  # 2  5
 
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
 
 
def show_batch():
    for epoch in range(2):  #  2  6
        for step, (batch_x, batch_y) in enumerate(loader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
 
 
if __name__ == '__main__':
    show_batch()