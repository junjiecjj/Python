queue=[0]*5
front=rear=-1
val=0
while rear<5 and val!=-1:
    val=int(input('请输入一个值加入队列，要从队列中取出值请输入0。(要结束则输入-1)：'))
    if val==0:
        if front==rear:
            print('[队列已经空了]')
            break
        front+=1
        if front==5:
            front=0
        print('从队列中取出值 [%d]' %queue[front])
        queue[front]=0
    elif val!=-1 and rear<5:
        if rear+1==front or rear==4 and front<=0:
            print('[队列已经满了]')
            break
        rear+=1
        if rear==5:
            rear=0
        queue[rear]=val
        
print('队列剩余数据：')
if front==rear:
    print('队列已空!!')
else:
    while front!=rear:
        front+=1
        if front==5:
            front=0
        print('[%d]' %queue[front],end='')
        queue[front]=0
print()