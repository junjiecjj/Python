def change(data):
    data[0],data[1]=data[1],data[0]
    print('函数内交换位置后：')
    for i in range(2):
        print('data[%d]=%3d' %(i,data[i]),end='\t')

#主程序
data=[16,25]
print('原始数据为：')
for i in range(2):
    print('data[%d]=%3d' %(i,data[i]),end='\t')
print('\n-------------------------------------')
change(data)
print('\n-------------------------------------')
print("排序后数据为：")
for i in range(2):
    print('data[%d]=%3d' %(i,data[i]),end='\t')