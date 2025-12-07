sum=0
number=int(input('请输入整数: '))
  
#递增for循环,从小到大打印出数字 
print('从小到大排列输出数字:')
for i in range(1,number+1):
    sum+=i  #设置sum为i的和 
    print('%d' %i,end='')
    #设置输出连加的算式 
    if i<number:
        print('+',end='')
    else:
        print('=',end='')
print('%d' %sum)

sum=0
#递减for循环,从大到小打印出数字 
print('从大到小排列输出数字:')
for i in range(number,0,-1):
    sum+=i 
    print('%d' %i,end='')
    if i<=1:
        print('=',end='')
    else:
        print('+',end='')
print('%d' %sum)