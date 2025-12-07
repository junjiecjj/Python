N=2
#声明2x2的数组arr并将所有元素赋值为 None
arr=[[None] * N for row in range(N)] 
print('|a1 b1|')
print('|a2 b2|')
arr[0][0]=input('请输入a1:')
arr[0][1]=input('请输入b1:')
arr[1][0]=input('请输入a2:')
arr[1][1]=input('请输入b2:')
#求二阶行列式的值
result = int(arr[0][0])*int(arr[1][1])-int(arr[0][1])*int(arr[1][0]) 
print('|%d %d|' %(int(arr[0][0]),int(arr[0][1])))
print('|%d %d|' %(int(arr[1][0]),int(arr[1][1])))
print('行列式值=%d' %result)