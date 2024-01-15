SIZE=8

def showdata(data):
    for i in range(SIZE):
        print('%3d' %data[i],end='')
    print()

def shell(data,size):
    k=1 #k打印计数
    jmp=size//2
    while jmp != 0:
        for i in range(jmp, size):  #i为扫描次数 jmp 为设置间距的位移量
            tmp=data[i]  #tmp用来暂存数据
            j=i-jmp      #以j来定位比较的元素
            while tmp<data[j] and j>=0:  #插入排序法
                data[j+jmp] = data[j]
                j=j-jmp
            data[jmp+j]=tmp
        print('第 %d 次排序过程：' %k,end='')  
        k+=1
        showdata (data)
        print('-----------------------------------------')
        jmp=jmp//2    #控制循环数

def main():
    data=[16,25,39,27,12,8,45,63]
    print('原始数组是：     ')	
    showdata (data)
    print('-----------------------------------------')
    shell(data,SIZE)

main()