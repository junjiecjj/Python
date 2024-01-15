class student:
    def __init__ (self):
        self.num=0
        self.name=''
        self.score=0
        self.next=None

print('请输入 5 项学生数据：')      
node=student()
if not node:
    print('[Error!!内存分配失败!]')
    sys.exit(0)
node.num=eval(input('请输入学号：'))
node.name=input('请输入姓名：')
node.score=eval(input('请输入成绩：'))
ptr=node #保留链表的头部，以ptr为当前节点指针
for i in range(1,5):
    newnode=student() #建立新节点
    if not newnode:
        print('[Error!!内存分配失败!')
        sys.exit(0)
    newnode.num=int(input('请输入学号：'))
    newnode.name=input('请输入姓名：')
    newnode.score=int(input('请输入成绩：'))
    newnode.next=None
    ptr.next=newnode #把新节点加在链表后面
    ptr=ptr.next  #让ptr保持在链表的最后面

print('  学  生  成  绩')
print(' 学号\t姓名\t成绩\n=====================')
ptr=node    #让ptr回到链表的头部
while ptr!=None:
    print('%3d\t%-s\t%3d' %(ptr.num,ptr.name,ptr.score))
    node=ptr
    ptr=ptr.next #ptr按序往后遍历链表