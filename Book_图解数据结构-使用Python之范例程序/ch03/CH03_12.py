class student:
    def __init_(self):
        self.name=''
        self.Math=0
        self.Eng=0
        self.no=''
        self.rlink=None
        self.llink=None

head=student()
head.llink=None
head.rlink=None
ptr=head #设置存取指针开始的位置
select=0
while True:
    select=int(input('(1)新增 (2)离开 =>'))
    if select==2:
        break;
    new_data=student()
    new_data.name=input('姓名: ')
    new_data.no=input('学号: ')
    new_data.Math=eval(input('数学成绩: '))
    new_data.Eng=eval(input('英语成绩: '))
    #输入节点结构中的数据 
    ptr.rlink=new_data
    new_data.rlink = None #下一个元素的next先设置为None
    new_data.llink=ptr #存取指针设置为新元素所在的位置
    ptr=new_data
        
ptr = head.rlink    #设置存取指针从链表头的右指针字段所指的节点开始
print()
while ptr!= None:
    print('姓名：%s\t学号:%s\t数学成绩:%d\t英语成绩:%d'  \
           %(ptr.name,ptr.no,ptr.Math,ptr.Eng))
    ptr = ptr .rlink  #将ptr移往右边下一个元素