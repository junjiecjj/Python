import random

val=0
data=[0]*80
for i in range(80):
    data[i]=random.randint(1,150)
while val!=-1:
    find=0
    val=int(input('请输入查找键值(1-150)，输入-1离开：'))
    for i in range(80):
        if data[i]==val:
            print('在第 %3d个位置找到键值 [%3d]' %(i+1,data[i]))
            find+=1
    if find==0 and val !=-1 :
        print('######没有找到 [%3d]######' %val)
print('数据内容为：')
for i in range(10):
    for j in range(8):
        print('%2d[%3d]  ' %(i*8+j+1,data[i*8+j]),end='')
    print('')