MAX=50
infix_q=['']*MAX

#运算符优先权的比较，若输入运算符小于堆栈中的运算符，
#则返回值为1，否则返回 0                          

#在中序法表达式和暂存堆栈中，运算符的优先级表，
#其优先权值为INDEX/2  
def compare(stack_o, infix_o):
    infix_priority=['']*9
    stack_priority=['']*8
    index_s=index_i=0
    infix_priority[0]='q'; infix_priority[1]=')'
    infix_priority[2]='+'; infix_priority[3]='-'
    infix_priority[4]='*'; infix_priority[5]='/'
    infix_priority[6]='^'; infix_priority[7]=' '
    infix_priority[8]='('
    
    stack_priority[0]='q'; stack_priority[1]='('
    stack_priority[2]='+'; stack_priority[3]='-'
    stack_priority[4]='*'; stack_priority[5]='/'
    stack_priority[6]='^'; stack_priority[7]=' '
    
    while stack_priority[index_s] != stack_o:
        index_s+=1

    while infix_priority[index_i] != infix_o:
        index_i+=1

    if int(index_s/2) >= int(index_i/2):
        return 1
    else:
        return 0
	
def infix_to_postfix():
    global MAX
    global infix_q
    rear=0; top=0; i=0
    #flag=0
    index = -1
    stack_t=['']*MAX  #以堆栈存储还不必输出的运算符

    str_=str(input('请开始输入中序法表达式: '))

    while i <len(str_):
        index+=1
        infix_q[index]=str_[i]
        i+=1

    infix_q[index+1]='q' #以q符号作为队列的结束符号
    
    print('后序法表达式: ', end='')
    stack_t[top]='q'  #在堆栈最底端加入q为结束符号

    for flag in range(index+2):
        if infix_q[flag]==')': #输入为)，则输出堆栈内的运算符，直到堆栈内为(
            while stack_t[top]!='(':
                print('%c' %stack_t[top],end='')
                top-=1
            top-=1
            #break
            #输入为q，则将堆栈内还未输出的运算符输出
        elif infix_q[flag]=='q':
            while stack_t[top]!='q':
                print('%c' %stack_t[top],end='')
                top -=1
            #break
            #输入为运算符，若小于TOP在堆栈中所指向的运算符，
            #则将Top所指向的运算符输出，若大于等于TOP在堆栈
            #中所指的运算符，则将输入的运算符压入堆栈
        elif infix_q[flag]=='(' or infix_q[flag]=='^' or \
             infix_q[flag]=='*' or infix_q[flag]=='/' or \
             infix_q[flag]=='+' or infix_q[flag]=='-' :
            
            while compare(stack_t[top], infix_q[flag])==1:
                print('%c' %stack_t[top], end='')
                top-=1
            top+=1
            stack_t[top] = infix_q[flag]
            #break
            #输入为操作数，则直接输出
        else:
            print('%c' %infix_q[flag],end='')
            #break

#主程序
print('------------------------------------------')
print('中序法表达式转成后序法表达式')
print('可以使用的运算符包括:^,*,+,-,/,(,)等 ')
print('------------------------------------------')

infix_to_postfix()