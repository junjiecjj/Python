def heap(data,size):
    for i in range(int(size/2),0,-1):#建立堆积树节点
        ad_heap(data,i,size-1)
    print()
    print('堆积的内容：',end='')
    for i in range(1,size): #原始堆积树的内容
        print('[%2d] ' %data[i],end='')
    print('\n')
    for i in range(size-2,0,-1): #堆积排序
        data[i+1],data[1]=data[1],data[i+1]#头尾节点交换
        ad_heap(data,1,i)#处理剩余节点
        print('处理过程为：',end='')
        for j in range(1,size):
            print('[%2d] ' %data[j],end='')
        print()

def ad_heap(data,i,size):
    j=2*i
    tmp=data[i]
    post=0
    while j<=size and post==0:
        if j<size:
            if data[j]<data[j+1]: #找出最大节点
                j+=1
        if tmp>=data[j]: #若树根较大，结束比较过程
            post=1
        else:
            data[int(j/2)]=data[j]#若树根较小，则继续比较
            j=2*j
    data[int(j/2)]=tmp #指定树根为父节点

def main():
    data=[0,5,6,4,8,3,2,7,1]	#原始数组的内容
    size=9
    print('原始数组为：',end='')
    for i in range(1,size):
        print('[%2d] ' %data[i],end='')
    heap(data,size) #建立堆积树
    print('排序结果为：',end='')
    for i in range(1,size):
        print('[%2d] ' %data[i],end='')
        
main()