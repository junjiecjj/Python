#用递归函数求 n 阶乘的值

def factorial(i):
    if i==0:
        return 1
    else:
        product = i * factorial(i-1) # sum=n*(n-1)!所以直接调用自身
        return product

n=int(input('请输入阶乘数:'))
for i in range(n+1):
    print('%d !值为 %3d' %(i,factorial(i)))