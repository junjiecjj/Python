class student:
    def __init__(self):
        self.name=''
        self.no=''
        self.next=None
          
head=student()  #新增链表头元素
ptr = head    #设置存取指针位置
ptr.next = None    #目前无一下个元素
select=0
while select!=2:
    select=int(input('(1)新增 (2)离开 =>'))
    if select ==2:
        break
    ptr.name=input('姓名 :')
    ptr.no=input('学号 :')
    new_data=student() #新增下一元素
    ptr.next=new_data   #连接下一元素
    new_data.next = None  #下一元素的next先设置为None
    ptr = new_data  #存取指针设置为新元素所在位置

ptr.next = head  #设置存取指针从头开始
print()
ptr=head

while True:
     print('姓名：%s\t学号:%s\n' %(ptr.name,ptr.no))
     ptr=ptr.next  #将head移往下一元素
     if ptr.next==head:
         break
print('---------------------------------------------------------')