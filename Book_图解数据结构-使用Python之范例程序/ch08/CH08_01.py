data=[16,25,39,27,12,8,45,63]	# 原始数据 
print('冒泡排序法：原始数据为：')
for i in range(8):
    print('%3d' %data[i],end='')
print()

for i in range(7,-1,-1): #扫描次数
    for j in range(i):
        if data[j]>data[j+1]:#比较,交换的次数
            data[j],data[j+1]=data[j+1],data[j]#比较相邻的两数,如果第一个数较大则交换
    print('第 %d 次排序后的结果是：' %(8-i),end='') #把各次扫描后的结果打印出来
    for j in range(8):
        print('%3d' %data[j],end='')
    print()
	
print('排序后的结果为：')
for j in range(8):
    print('%3d' %data[j],end='')
print()