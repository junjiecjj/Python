SIZE=8        #定义数组大小
def showdata(data):
    for i in range(SIZE):
        print('%3d' %data[i],end='')   #打印数组数据
    print()
    
def insert(data):
    for i in range(1,SIZE):
        tmp=data[i] #tmp用来暂存数据
        no=i-1
        while no>=0 and tmp<data[no]:
            data[no+1]=data[no]	#就把所有元素往后推一个位置
            no-=1
        data[no+1]=tmp #最小的元素放到第一个位置

def main():
    data=[16,25,39,27,12,8,45,63]
    print('原始数组是：')
    showdata(data)
    insert(data)
    print('排序后的数组是：')
    showdata(data)

main()