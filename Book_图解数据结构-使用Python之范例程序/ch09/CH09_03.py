import random

def interpolation_search(data,val):
    low=0
    high=49
    print('查找过程中......')
    while low<= high and val !=-1:
        mid=low+int((val-data[low])*(high-low)/(data[high]-data[low])) #插值查找法公式
        if val==data[mid]:
            return mid
        elif val < data[mid]:
            print('%d 介于位置 %d[%3d] 和中间值 %d[%3d] 之间，找左半边' \
                  %(val,low+1,data[low],mid+1,data[mid]))
            high=mid-1
        elif val > data[mid]:
            print('%d 介于中间值位置 %d[%3d] 和 %d[%3d] 之间，找右半边' \
                  %(val,mid+1,data[mid],high+1,data[high]))
            low=mid+1
    return -1

val=1
data=[0]*50
for i in range(50):
    data[i]=val
    val=val+random.randint(1,5)

while True:
    num=0
    val=int(input('请输入查找键值(1-150)，输入-1结束：'))
    if val==-1:
        break
    num=interpolation_search(data,val)
    if num==-1:
         print('##### 没有找到[%3d] #####' %val)
    else:
        print('在第 %2d个位置找到 [%3d]' %(num+1,data[num]))
	
print('数据内容为：')
for i in range(5):
    for j in range(10):
        print('%3d-%-3d' %(i*10+j+1,data[i*10+j]),end='')
    print()