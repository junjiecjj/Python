A= [[1,3,5],[7,9,11],[13,15,17]] #二维数组的声明
B= [[9,8,7],[6,5,4],[3,2,1]]     #二维数组的声明
N=3
C=[[None] * N for row in range(N)]
	
for i in range(3):
    for j in range(3):
        C[i][j]=A[i][j]+B[i][j] #矩阵C = 矩阵A + 矩阵B
print('[矩阵A和矩阵B相加的结果]') #打印出A+B的内容
for i in range(3):
    for j in range(3):
        print('%d' %C[i][j], end='\t')
    print()