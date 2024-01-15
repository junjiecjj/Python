def showdata (data):
    for i in range(8):
        print('%3d' %data[i],end='')
    print()

def select (data):
    for i in range(7):
        for j in range(i+1,8):
            if data[i]>data[j]: #比较第i个和第j个元素
                data[i],data[j]=data[j],data[i]
    print()

data=[16,25,39,27,12,8,45,63]
print('原始数据为：')
for i in range(8):
    print('%3d' %data[i],end='')
print('\n-------------------------------------')
select(data)
print("排序后的数据为：")
for i in range(8):
    print('%3d' %data[i],end='')
print('')