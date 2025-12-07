product=1
i=1
while i<6:
    product=i*product
    print('i=%d' %i,end='') 
    print('\tproduct=%d' %product)
    i+=1
print('\n阶乘的结果=%d'%product)
print()