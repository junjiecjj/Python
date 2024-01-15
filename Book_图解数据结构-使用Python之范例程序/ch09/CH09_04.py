MAX=20

def fib(n):
    if n==1 or n==0:
        return n
    else:
        return fib(n-1)+fib(n-2)
    
def fib_search(data,SearchKey):
    global MAX
    index=2
    #斐波拉契数列的查找
    while fib(index)<=MAX :
        index+=1
    index-=1
    # index >=2
    #起始的斐波拉契数
    RootNode=fib(index)
    #前一个斐波拉契数
    diff1=fib(index-1)
    #前两个斐波拉契数即diff2=fib(index-2)
    diff2=RootNode-diff1
    RootNode-=1 #这个表达式是配合数组的索引，是从0开始存储数据的
    while True:
        if SearchKey==data[RootNode]:
            return RootNode
        else:
            if index==2:
                return MAX #没有找到
            if SearchKey<data[RootNode]:
                RootNode=RootNode-diff2  #左子树的新斐波拉契数
                temp=diff1
                diff1=diff2      #前一个斐波拉契数
                diff2=temp-diff2 #前两个斐波拉契数
                index=index-1
            else:
                if index==3:
                    return MAX
                RootNode=RootNode+diff2  #右子树的新斐波拉契数
                diff1=diff1-diff2  #前一个斐波拉契数
                diff2=diff2-diff1  #前两个斐波拉契数
                index=index-2
                

data=[5,7,12,23,25,37,48,54,68,77, \
      91,99,102,110,118,120,130,135,136,150]
i=0
j=0
while True:
    val=int(input('请输入查找键值(1-150)，输入-1结束：'))
    if val==-1: #输入值为-1就跳离循环
        break
    RootNode=fib_search(data,val)#使用斐波拉契查找法查找数据
    if RootNode==MAX:
        print('##### 没有找到[%3d] #####' %val)
    else:
        print('在第 %2d个位置找到 [%3d]' %(RootNode+1,data[RootNode]))
   
print('数据内容为：')
for i in range(2):
    for j in range(10):
        print('%3d-%-3d' %(i*10+j+1,data[i*10+j]),end='')
    print()