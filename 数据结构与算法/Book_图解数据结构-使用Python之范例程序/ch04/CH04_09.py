global queen
global number
EIGHT=8 #定义堆栈的最大容量
queen=[None]*8 #存放8个皇后的行位置

number=0    #计算总共有几组解的总数
#决定皇后存放的位置
#输出所需要的结果
def print_table():
    global number
    x=y=0
    number+=1
    print('')
    print('八皇后问题的第%d组解\t' %number)
    for x in range(EIGHT):
        for y in range(EIGHT):
            if x==queen[y]:
                print('<q>',end='')
            else:
                print('<->',end='')
        print('\t')
    input('\n..按下任意键继续..\n')

#测试在(row,col)上的皇后是否遭受攻击
#若遭受攻击则返回值为1,否则返回0
def attack(row,col):
    global queen
    i=0
    atk=0
    offset_row=offset_col=0
    while (atk!=1)and i<col:
        offset_col=abs(i-col)
        offset_row=abs(queen[i]-row)
        #判断两皇后是否在同一行或在同一对角线上
        if queen[i]==row or offset_row==offset_col:
            atk=1
        i=i+1
    return atk

def decide_position(value):
    global queen
    i=0
    while i<EIGHT:
        if attack(i,value)!=1:
            queen[value]=i
            if value==7:
                print_table()
            else:
                decide_position(value+1)
        i=i+1

#主程序
decide_position(0)