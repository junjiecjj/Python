class student:
    def __init__(self):
        self.name=' '*20
        self.score=0
        self.next=None
        
front=student()
rear=student()
front=None
rear=None

def enqueue(name, score):  # 把数据加入队列
    global front
    global rear
    new_data=student()  # 分配内存给新元素
    new_data.name=name  # 给新元素赋值
    new_data.score = score
    if rear==None:      # 如果rear为None，表示这是第一个元素
        front = new_data
    else:
        rear.next = new_data    # 将新元素连接到队列末尾

    rear = new_data         # 将rear指向新元素，这是新的队列末尾
    new_data.next = None    # 新元素之后无其他元素

def dequeue(): # 取出队列中的数据
    global front
    global rear
    if front == None:
        print('队列已空！')
    else:
        print('姓名：%s\t成绩：%d ....取出' %(front.name, front.score))
        front = front.next    # 将队列前端移到下一个元素
        
def show():     # 显示队列中的数据
    global front
    global rear
    ptr = front
    if ptr == None:
        print('队列已空！')
    else:
        while ptr !=None: # 从front到rear遍历队列
            print('姓名：%s\t成绩：%d' %(ptr.name, ptr.score))
            ptr = ptr.next

select=0
while True:
    select=int(input('(1)加入 (2)取出 (3)显示 (4)离开 => '))
    if select==4:
        break
    if select==1:
        name=input('姓名: ')
        score=int(input('成绩: '))
        enqueue(name, score)
    elif select==2:
        dequeue()
    else:
        show()