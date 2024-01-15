import sys

class student:
    def __init__(self):
        self.name=''
        self.Math=0
        self.Eng=0
        self.no=''
        self.next=None
        
head=student() #建立链表的头部
head.next=None
ptr = head
Msum=Esum=num=student_no=0
select=0

while select !=2:
    print('(1)新增 (2)离开 =>')
    try:
        select=int(input('请输入一个选项: '))
    except ValueError:
         print('输入错误')
         print('请重新输入\n')
    if select ==1:
        new_data=student() #新增下一个元素
        new_data.name=input('姓名:')
        new_data.no=input('学号:')
        new_data.Math=eval(input('数学成绩:'))
        new_data.Eng=eval(input('英语成绩:'))
        ptr.next=new_data #存取指针设置为新元素所在的位置
        new_data.next=None #下一元素的next先设置为None
        ptr=ptr.next
        num=num+1

ptr=head.next #设置存取指针从链表的头部开始
print()
while ptr !=None:
    print('姓名：%s\t学号:%s\t数学成绩:%d\t英语成绩:%d' \
           %(ptr.name,ptr.no,ptr.Math,ptr.Eng))
    Msum=Msum+ptr.Math
    Esum=Esum+ptr.Eng
    student_no=student_no+1
    ptr=ptr.next #将ptr移往下一个元素

if student_no !=0:
    print('---------------------------------------------------------')
    print('本链表中学生的数学平均成绩:%.2f 英语平均成绩:%.2f' \
      %(Msum/student_no,Esum/student_no))