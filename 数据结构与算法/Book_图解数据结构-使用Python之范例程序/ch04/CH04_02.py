import random
global top

top=-1
k=0

def push(stack,MAX,val):
    global top
    if top>=MAX-1:
        print('[堆栈已经满了]')
    else:
        top=top+1
        stack[top]=val
        
def pop(stack):
    global top
    if top<0:
        print('[堆栈已经空了]')
    else:
        top=top-1
        return stack[top]

def shuffle(old):
    result=[]
    while old:
        p=random.randrange(0,len(old))
        result.append(old[p])
        old.pop(p)
    return result

card=[None]*52
card_new=[None]*52
stack=[0]*52
for i in range(52):
    card[i]=i+1

print('[洗牌中...请稍候!]')

card_new=shuffle(card)

i=0
while i!=52:
    push(stack,52,card_new[i])  #将52张牌压入堆栈
    i=i+1

print('[逆时针发牌]')
print('[显示各家的牌] 东家\t  北家\t   西家\t    南家')
print('=================================')

while top>=0:
    #print(stack[top])
    style = (stack[top]) % 4	#计算牌的花色
    #print('style=', style)
    if style==0:  #梅花
        ascVal='club'
    elif style==1:  #方块
        ascVal='diamond'
    elif style==2:   #红心
        ascVal='heart'
    elif style==3:
        ascVal='spade'   #黑桃
    
    print('[%s%3d]\t' %(ascVal,stack[top]%13+1),end='')
    if top%4==0:
        print()
    top-=1