MAXSTACK=100 #定义堆栈的最大容量
global stack
stack=[None]*MAXSTACK  #堆栈的数组声明
top=-1 #堆栈的顶端

#判断是否为空堆栈
def isEmpty():
    if top==-1:
        return True
    else:
        return False

#将指定的数据压入堆栈
def push(data):
    global top
    global MAXSTACK
    global stack
    if top>=MAXSTACK-1:
        print('堆栈已满,无法再加入')
    else:
        top +=1
        stack[top]=data #将数据压入堆栈

#从堆栈弹出数据*/
def pop():
    global top
    global stack
    if isEmpty():
        print('堆栈是空')
    else:
        print('弹出的元素为: %d' % stack[top])
        top=top-1     
        
#主程序
i=2
count=0
while True:
    i=int(input('要压入堆栈,请输入1,要弹出则输入0,停止操作则输入-1: '))
    if i==-1:
        break
    elif i==1:
        value=int(input('请输入元素值:'))
        push(value)
    elif i==0:
        pop()

print('============================')
if top <0:
    print('\n 堆栈是空的')
else:
    i=top
    while i>=0:
        print('堆栈弹出的顺序为:%d' %(stack[i]))
        count +=1
        i =i-1
    print 

print('============================')  