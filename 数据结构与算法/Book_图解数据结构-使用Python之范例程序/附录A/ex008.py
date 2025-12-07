sum=0
number=1
while True:
    if number==0:
        break
    number=int(input('数字0为结束程序,请输入数字: '))
    sum+=number
    print('目前累加的结果为: %d' %sum)